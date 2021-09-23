import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import hyperparameters as hparams

class SqueezeExcitation(nn.Module):
    def __int__(self):
        super(SqueezeExcitation, self).__init__()

    def forward(self, x):

class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.se1 = SqueezeExcitation()
        self.se2 = SqueezeExcitation()
        self.se3 = SqueezeExcitation()

        self.conv1 = torch.nn.Conv2d()

    def forward(self, x):
        # x -> (B, C, H, W)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.recognizer = Recognizer()
        self.hparams =

    def forward(self, x):

if __name__ == '__main__':
    model = Model()