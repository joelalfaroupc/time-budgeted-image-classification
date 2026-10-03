import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_best_checkpoint_does_not_follow_later_updates(tmp_path):
    from model import ElVostreModel
    from train import save_best

    model = ElVostreModel(2)
    initial = model.fc2.weight.detach().clone()
    destination = tmp_path / 'best.pth'
    save_best(model, destination, ['a', 'b'], 1, 0.8)
    with torch.no_grad():
        model.fc2.weight.add_(10)
    stored = torch.load(destination, weights_only=True)
    assert torch.equal(stored['state_dict']['fc2.weight'], initial)
    assert stored['classes'] == ['a', 'b']
    assert stored['epoch'] == 1


def test_original_checkpoint_has_twelve_finite_outputs():
    from model import load_model

    model, classes = load_model(ROOT / 'checkpoints/best_model.pth')
    assert classes is None
    assert sum(p.numel() for p in model.parameters()) == 457676
    with torch.no_grad():
        output = model(torch.zeros(2, 1, 28, 28))
    assert output.shape == (2, 12)
    assert torch.isfinite(output).all()


def test_wrong_class_order_is_rejected():
    from train import validate_classes

    with pytest.raises(ValueError, match='class'):
        validate_classes(['a', 'b'], ['b', 'a'])


def test_cli_train_and_predict_on_small_image_folder(tmp_path):
    for split in ['train', 'validation']:
        for name, color in [('a', 30), ('b', 220)]:
            folder = tmp_path / 'data' / split / name
            folder.mkdir(parents=True)
            for i in range(3):
                Image.new('L', (28, 28), color + i).save(folder / f'{i}.png')
    output = tmp_path / 'run'
    subprocess.run([sys.executable, str(ROOT / 'train.py'), '--data-dir',
                    str(tmp_path / 'data'), '--output', str(output),
                    '--epochs', '2', '--batch-size', '4', '--workers', '0',
                    '--time-budget', '30', '--threads', '1'], check=True, timeout=60)
    checkpoint = torch.load(output / 'best.pth', weights_only=True)
    assert checkpoint['classes'] == ['a', 'b']
    assert 1 <= checkpoint['epoch'] <= 2
    assert len((output / 'metrics.csv').read_text().splitlines()) == 3
    prediction = subprocess.run([sys.executable, str(ROOT / 'predict.py'),
                                 str(tmp_path / 'data/validation/a/0.png'),
                                 '--checkpoint', str(output / 'best.pth')],
                                check=True, capture_output=True, text=True, timeout=30)
    result = json.loads(prediction.stdout)
    assert result['predictions'][0]['label'] in ['a', 'b']
    assert abs(sum(p['probability'] for p in result['predictions']) - 1) < 1e-5


def test_zero_budget_does_not_write_untrained_checkpoint(tmp_path):
    from argparse import Namespace
    from train import validate_options

    with pytest.raises(ValueError, match='positive'):
        validate_options(Namespace(time_budget=0, epochs=2, batch_size=4, workers=0,
                                   threads=1, image_size=None))
