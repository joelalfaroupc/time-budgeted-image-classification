"""The submitted CNN architecture, with its original checkpoint key names."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ElVostreModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = self.block(1, 32)
        self.conv2 = self.block(32, 64)
        self.conv3 = self.block(64, 128)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)

    @staticmethod
    def block(inputs, outputs):
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.BatchNorm2d(outputs), nn.LeakyReLU(0.01), nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        x = x.view(x.size(0), -1)
        return self.fc2(self.dropout(F.leaky_relu(self.fc1(x), 0.01)))


def read_checkpoint(path):
    """Read tensor/primitive data without unpickling arbitrary Python objects."""
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict'], checkpoint
    return checkpoint, {}


def load_model(path):
    state, metadata = read_checkpoint(path)
    model = ElVostreModel(state['fc2.weight'].shape[0])
    model.load_state_dict(state, strict=True)
    classes = metadata.get('classes')
    if classes is not None and (len(classes) != model.fc2.out_features
                               or len(set(classes)) != len(classes)):
        raise ValueError('Checkpoint class mapping does not match the output layer.')
    return model.eval(), classes
