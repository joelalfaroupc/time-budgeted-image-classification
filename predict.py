"""Classify one image using the supplied weights or a newly trained checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from model import load_model, read_checkpoint
from train import make_transforms


def run(args):
    torch.set_num_threads(args.threads)
    model, classes = load_model(args.checkpoint)
    _, metadata = read_checkpoint(args.checkpoint)
    if args.classes:
        supplied = json.loads(args.classes.read_text())
        if not isinstance(supplied, list) or not all(isinstance(x, str) for x in supplied):
            raise ValueError('Classes must be a JSON list of names in training class-index order.')
        if len(supplied) != model.fc2.out_features or len(set(supplied)) != len(supplied):
            raise ValueError('Class mapping does not match the checkpoint output layer.')
        if classes is not None and classes != supplied:
            raise ValueError('Class mapping conflicts with the checkpoint metadata.')
        classes = supplied
    trained_size = metadata.get('image_size')
    if trained_size is not None and args.image_size is not None and args.image_size != trained_size:
        raise ValueError('Image size conflicts with the checkpoint preprocessing metadata.')
    size = args.image_size if args.image_size is not None else trained_size
    _, transform = make_transforms(size)
    with Image.open(args.image) as image:
        if min(image.size) < 8 and size is None:
            raise ValueError('Images must be at least 8 pixels per side, or specify --image-size.')
        tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(tensor).softmax(1)[0]
    predictions = [{'class_index': index,
                    'label': classes[index] if classes else f'class_{index}',
                    'probability': float(probabilities[index])}
                   for index in probabilities.argsort(descending=True).tolist()]
    print(json.dumps({'image': str(args.image), 'class_names_known': classes is not None,
                      'predictions': predictions}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/best_model.pth'))
    parser.add_argument('--classes', type=Path, default=None)
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads <= 0 or (args.image_size is not None and args.image_size < 8):
        parser.error('Threads must be positive and image size must be at least 8.')
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
