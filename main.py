import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperparameters as hparams

class SqueezeExcitation(nn.Module):
    def __init__(self):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = torch.nn.Linear(hparams.input_channel, hparams.input_channel/hparams.r)
        self.fc2 = torch.nn.Linear(hparams.input_channel/hparams.r, hparams.input_channel)
        self.sm = torch.nn.Sigmoid()

    def forward(self, x):
        # x -> (B, C, H, W)
        _, _, H, W = x.shape
        output = x.clone()
        x = torch.sum(x, axis=(2,3)) / (H*W)
        x = x.squeeze()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sm(x)
        x = x.unsqueeze(2).unsqueeze(3)
        return output * x


class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.acs = SqueezeExcitation()
        self.se1 = SqueezeExcitation()
        self.se2 = SqueezeExcitation()
        self.se3 = SqueezeExcitation()

        self.conv1 = torch.nn.Conv2d(hparams.input_channel,
                                     hparams.output_channel,
                                     kernel_size=4,
                                     stride=2,
                                     padding=1)
        self.conv2 = torch.nn.Conv2d(hparams.output_channel,
                                     hparams.output_channel,
                                     kernel_size=4,
                                     stride=4,
                                     padding=1)
        self.conv3 = torch.nn.Conv2d(hparams.output_channel,
                                     hparams.output_channel,
                                     kernel_size=4,
                                     stride=4)
        self.fc = torch.nn.Linear(4,1)
        self.sm = torch.nn.Sigmoid()

    def forward(self, x):
        # x -> (B, C, H, W)
        B, C, _, _ = x.shape
        sp = self.acs(x)
        x = self.conv1(sp)
        x = self.se1(x)
        x = self.conv2(x)
        x = self.se2(x)
        x = self.conv3(x)
        x = self.se3(x)
        # x -> (B, C, 2, 2)
        x = x.reshape(B,C,4)
        x = self.fc(x)
        # x -> (B, C, 1)
        return sp, self.sm(x).squeeze()



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.recognizer = Recognizer()
        self.criterion = torch.nn.BCELoss()

    def forward(self, x):
        sparse, bce = self.recognizer(x)
        target = torch.empty(bce)
        loss = self.criterion(bce, target)
        loss += torch.norm(sparse, 1)
        return loss


if __name__ == '__main__':
    model = Model()