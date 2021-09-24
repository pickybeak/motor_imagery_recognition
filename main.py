import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperparameters as hparams
import torch.optim as optim

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channel):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = torch.nn.Linear(input_channel, int(input_channel/hparams.r))
        self.fc2 = torch.nn.Linear(int(input_channel/hparams.r), input_channel)
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
        self.acs = SqueezeExcitation(hparams.input_channel)
        self.se1 = SqueezeExcitation(hparams.output_channel)
        self.se2 = SqueezeExcitation(hparams.output_channel)
        self.se3 = SqueezeExcitation(hparams.output_channel)

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
                                     stride=4,
                                     padding=1)
        self.fc = torch.nn.Linear(4,1)
        self.sm = torch.nn.Sigmoid()

    def forward(self, x):
        # x -> (B, C, H, W)
        B, _, _, _ = x.shape
        sp = self.acs(x)
        x = self.conv1(sp)
        x = self.se1(x)
        x = self.conv2(x)
        x = self.se2(x)
        x = self.conv3(x)
        x = self.se3(x)
        # x -> (B, C, 2, 2)
        x = x.reshape(B,-1,4)
        x = self.fc(x)
        # x -> (B, C, 1)
        return sp, self.sm(x).squeeze()



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.recognizer = Recognizer()
        # self.criterion = torch.nn.BCELoss()

    def forward(self, x):
        sparse, bce = self.recognizer(x)
        # target = torch.empty(bce)
        # loss = self.criterion(bce, target)
        sparse_loss = torch.norm(sparse, 1)
        return bce, sparse_loss

dataset = [(torch.randn(3,22,64,64), torch.zeros(3,64))]

if __name__ == '__main__':
    model = Model()
    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(2):
        running_loss = 0.0
        for data in dataset:
            i = 0
            # load data
            input, label = data
            optimizer.zero_grad()
            bce, sparse_loss = model(input)
            loss = criterion(bce, label)
            loss += sparse_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1 == 0:
                print('epoch%d, step%5d loss: %.3f'%(epoch+1, i+1, running_loss/1))
                running_loss = 0

    print('train finished')