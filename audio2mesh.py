
import torch
from torch import nn
from torch.nn import functional as F
import math


class Audio2mesh(nn.Module):
    '''
    This is a PyTorch model class representing Audio Encoder, and a geometry decoder mentioned in the paper
    '''
    def __init__(self):
        super(Audio2mesh, self).__init__()

        # TODO: Complete the following audio encoder layers as mentioned in the paper
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(2, 72, (3, 1), (2, 1), (1, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(72, 108, (3, 1), (2, 1), (1, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(108, 162, (3, 1), (2, 1), (1, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(162, 243, (3, 1), (2, 1), (1, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(243, 256, (3, 1), (2, 1), (1, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (3, 1), (2, 1), (1, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, (1, 3), (1, 2), (0, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, (1, 3), (1, 2), (0, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, (1, 3), (1, 2), (0, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, (1, 3), (1, 2), (0, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, (1, 3), (1, 2), (0, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(8, 4, (1, 3), (1, 2), (0, 1)),
            nn.LeakyReLU(),
            View([-1, 32]),


        ,)
        # TODO: Complete the following geometry decoder mentioned in the paper
        self.geometry_decoder = nn.Sequential(
            nn.Linear(32, 150),
            nn.Dropout(0.5),
            nn.Linear(150, 1404)
        ,)

    def forward(self, audio_spectogram, verticref):
        '''
        Forward pass of model
        Input: Audio Spectogram of shape (256,24,2)
        Returns: Vertices of shape (1404) ----> 468*3

        '''
        # TODO: Pass the audio spectogram though the audio encoder and geometry decoder
        proj = self.audio_encoder(audio_spectogram)
        vertices = self.geometry_decoder(proj)
        vertices = vertices.reshape(-1, 468, 3)
        vertices_mod = verticref + vertices

        return vertices_mod
