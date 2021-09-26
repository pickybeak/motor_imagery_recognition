import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import hyperparameters as hparams
import torch.optim as optim
import pickle
import math

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channel):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = torch.nn.Linear(input_channel, int(input_channel/hparams.r))
        self.fc2 = torch.nn.Linear(int(input_channel/hparams.r), input_channel)
        self.sm = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x -> (B, C, H, W)
        _, _, H, W = x.shape
        output = x.clone()
        x = torch.sum(x, axis=(2,3)) / (H*W)
        x = x.squeeze()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sm(x)
        x = x.unsqueeze(2).unsqueeze(3)
        return output * x, x


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
        self.elu1 = torch.nn.ELU()
        self.conv2 = torch.nn.Conv2d(hparams.output_channel,
                                     hparams.output_channel,
                                     kernel_size=4,
                                     stride=4)
        self.elu2 = torch.nn.ELU()
        self.conv3 = torch.nn.Conv2d(hparams.output_channel,
                                     hparams.output_channel,
                                     kernel_size=4,
                                     stride=4)
        self.elu3 = torch.nn.ELU()
        self.fc = torch.nn.Linear(4,1)
        self.ffc = torch.nn.Linear(64,1)
        self.sm = torch.nn.Sigmoid()

    def forward(self, x):
        # x -> (B, C, H, W)
        B, _, _, _ = x.shape
        x, sp = self.acs(x)
        x = self.conv1(x)
        x = self.elu1(x)
        x, _ = self.se1(x)
        x = self.conv2(x)
        x = self.elu2(x)
        x, _ = self.se2(x)
        x = self.conv3(x)
        x = self.elu3(x)
        x, _ = self.se3(x)
        # x -> (B, C, 2, 2)
        x = x.reshape(B,-1,4)
        x = self.fc(x).squeeze()
        x = self.ffc(x)
        # x -> (B, C, 1)
        return sp, self.sm(x).squeeze()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.recognizer = Recognizer()
        # self.criterion = torch.nn.BCELoss()

    def forward(self, x):
        B, _, _, _ = x.shape
        sparse, bce = self.recognizer(x)
        # target = torch.empty(bce)
        # loss = self.criterion(bce, target)
        sparse_loss = torch.norm(sparse, 1) / B
        return bce, sparse_loss

if __name__ == '__main__':
    train_acc_by_subject = []
    val_acc_by_subject = []

    for i in range(1, 2):
        file = 'A0%d_64x64_scipy.pkl'%i
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        # plt.subplot(3,3,i)
        train_loss = []
        val_loss = []

        model = Model()
        criterion = torch.nn.BCELoss()
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)

        for epoch in range(hparams.epoch):
            running_loss = 0.0
            # assert dataset['X_train'].shape[0] % hparams.batch_size == 0
            per_batch = math.ceil(dataset['X_train'].shape[0] / hparams.batch_size)
            for batch_index in range(hparams.batch_size):
                # load data
                running_loss = 0.0
                input = torch.Tensor(dataset['X_train'][batch_index:(per_batch*(batch_index+1))])
                label = torch.Tensor(dataset['y_train'][batch_index:(per_batch*(batch_index+1))])
                optimizer.zero_grad()
                bce, sparse_loss = model(input)
                y_pred = bce > 0.5
                loss = criterion(bce, label)
                loss += sparse_loss
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if batch_index % 1 == 0:
                    print('subject%d' % i)
                    print('epoch%d, batch%d train_loss: %.3f'%(epoch+1, batch_index+1, running_loss/1))
                    print('train_accuracy: %.3f' % ((torch.sum(y_pred == label) / y_pred.shape[0])*100))

            train_loss.append(running_loss)
            with torch.no_grad():
                input = torch.Tensor(dataset['X_test'])
                label = torch.Tensor(dataset['y_test'])
                bce, sparse_loss = model(input)
                y_pred = bce > 0.5
                loss = criterion(bce, label)
                loss += sparse_loss
                val_loss.append(loss)
                print('------------------------epoch test loss: %.3f' % (loss))
                print('------------------------epoch test accuracy: %.3f' % ((torch.sum(y_pred == label) / y_pred.shape[0]) * 100))

        with torch.no_grad():
            input = torch.Tensor(dataset['X_train'])
            label = torch.Tensor(dataset['y_train'])
            bce, sparse_loss = model(input)
            y_pred = bce > 0.5
            train_acc = (torch.sum(y_pred == label) / y_pred.shape[0]) * 100
            train_acc_by_subject.append(float(train_acc))
            print('------------------------total train accuracy: %.3f' % train_acc)

            input = torch.Tensor(dataset['X_test'])
            label = torch.Tensor(dataset['y_test'])
            bce, sparse_loss = model(input)
            y_pred = bce > 0.5
            val_acc = (torch.sum(y_pred == label) / y_pred.shape[0]) * 100
            val_acc_by_subject.append(float(val_acc))
            print('------------------------total test accuracy: %.3f' % val_acc)

        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.legend(['train_loss', 'test_loss'], loc="upper right")

    print('train finished')
    print('train_acc', train_acc_by_subject)
    print('val_acc', val_acc_by_subject)
    plt.show()