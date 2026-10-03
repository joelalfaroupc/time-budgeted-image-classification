"""Train the course CNN on ImageFolder splits with a bounded training loop."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ElVostreModel


def validate_options(args):
    if min(args.time_budget, args.epochs, args.batch_size, args.threads) <= 0:
        raise ValueError('Time budget, epochs, batch size and threads must be positive.')
    if args.workers < 0:
        raise ValueError('Workers must be nonnegative.')
    if args.image_size is not None and args.image_size < 16:
        raise ValueError('Training image size must be at least 16 pixels.')


def validate_classes(train_classes, validation_classes):
    if train_classes != validation_classes:
        raise ValueError('Train and validation class names and order must match.')


def preprocessing(image_size=None):
    return ([transforms.Resize((image_size, image_size))] if image_size else [])


def make_transforms(image_size=None, horizontal_flip=True):
    validation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), *preprocessing(image_size),
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
    ])
    training = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), *preprocessing(image_size),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomHorizontalFlip(p=0.5 if horizontal_flip else 0),
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
    ])
    return training, validation


def save_best(model, destination, classes, epoch, accuracy, image_size=None):
    # Materialize the snapshot now; state_dict() alone aliases live parameters.
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save({'state_dict': state, 'classes': list(classes), 'epoch': epoch,
                'accuracy': float(accuracy), 'image_size': image_size}, destination)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    correct, count, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        scores = model(images)
        loss_sum += criterion(scores, labels).item() * len(labels)
        correct += (scores.argmax(1) == labels).sum().item()
        count += len(labels)
    return loss_sum / count, correct / count


def run(args):
    validate_options(args)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    training_transform, validation_transform = make_transforms(
        args.image_size, not args.no_horizontal_flip)
    training = datasets.ImageFolder(args.data_dir / 'train', training_transform)
    validation = datasets.ImageFolder(args.data_dir / 'validation', validation_transform)
    validate_classes(training.classes, validation.classes)
    # Without an explicit resize, each batch must contain equal-sized images.
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers)
    model = ElVostreModel(len(training.classes))  # CPU, as in the competition submission.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, threshold=0.001, cooldown=1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'config.json').write_text(json.dumps({
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'classes': training.classes, 'torch': torch.__version__,
    }, indent=2) + '\n')
    start = time.monotonic()
    best, epochs_completed = -1.0, 0
    with (args.output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'epoch', 'train_loss', 'validation_loss', 'validation_accuracy',
            'learning_rate', 'training_samples', 'complete_epoch', 'elapsed_seconds'])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            samples, loss_sum = 0, 0.0
            for images, labels in train_loader:
                if time.monotonic() - start >= args.time_budget:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                samples += len(labels)
                loss_sum += loss.item() * len(labels)
            if not samples:
                break
            val_loss, accuracy = evaluate(model, val_loader, criterion)
            learning_rate = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            if accuracy > best:
                best = accuracy
                save_best(model, args.output / 'best.pth', training.classes,
                          epoch, accuracy, args.image_size)
            writer.writerow(dict(epoch=epoch, train_loss=loss_sum / samples,
                                 validation_loss=val_loss, validation_accuracy=accuracy,
                                 learning_rate=learning_rate, training_samples=samples,
                                 complete_epoch=samples == len(training),
                                 elapsed_seconds=time.monotonic() - start))
            handle.flush()
            epochs_completed += 1
            print(f'Epoch {epoch}: validation accuracy {accuracy:.2%}, loss {val_loss:.4f}', flush=True)
            if time.monotonic() - start >= args.time_budget:
                break
    if not epochs_completed:
        raise RuntimeError('No training batch completed; increase the time budget.')
    print(f'Best validation accuracy: {best:.2%}. Checkpoint: {args.output / "best.pth"}')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data-dir', type=Path, default=Path('data'))
    result.add_argument('--output', type=Path, default=Path('runs/experiment'))
    result.add_argument('--time-budget', type=float, default=600)
    result.add_argument('--epochs', type=int, default=1000)
    result.add_argument('--batch-size', type=int, default=32)
    result.add_argument('--workers', type=int, default=4)
    result.add_argument('--threads', type=int, default=4)
    result.add_argument('--seed', type=int, default=123)
    result.add_argument('--image-size', type=int, default=None)
    result.add_argument('--no-horizontal-flip', action='store_true')
    return result


if __name__ == '__main__':
    arguments = parser().parse_args()
    try:
        run(arguments)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
