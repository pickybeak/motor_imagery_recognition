import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import hyperparameters as hparams
import torch.optim as optim
import pickle
import math
import statistics
import time

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
	
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channel):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = torch.nn.Linear(input_channel, int(input_channel/hparams.r), bias=False)
        self.fc2 = torch.nn.Linear(int(input_channel/hparams.r), input_channel, bias=False)
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
        self.elu4 = torch.nn.ELU()
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
        x = self.elu4(x)
        x = self.ffc(x)
        # x -> (B, C, 1)
        return sp, self.sm(x).squeeze()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.recognizer = Recognizer()

    def forward(self, x):
        B, _, _, _ = x.shape
        sparse, bce = self.recognizer(x)
        sparse_loss = torch.norm(sparse, 1) / B
        return bce, sparse_loss

if __name__ == '__main__':

    cuda = torch.device('cuda:1')
    # 9 subject, 10 fold, 500 epoch
    subject = 9
    fold = 10
    total_train_loss, total_train_acc = torch.empty(subject, fold, hparams.epoch).to(cuda), torch.empty(subject, fold, hparams.epoch).to(cuda),
    total_val_loss, total_val_acc = torch.empty(subject, fold, hparams.epoch).to(cuda), torch.empty(subject, fold, hparams.epoch).to(cuda),

    for i in range(subject):
        file = 'A0%d_64x64_scipy2_cv10.pkl'%(i+1)
        with open(file, "rb") as f:
            dataset = pickle.load(f)
            X = dataset["X"]
            Y = dataset["y"]
            folds = dataset["folds"]

        X = torch.Tensor(X).to(cuda)
        Y = torch.Tensor(Y).to(cuda)

        # plt.subplot(3,3,i)

        for f, (train_index, test_index) in enumerate(folds):

            model = Model().to(cuda)
            criterion = torch.nn.BCELoss()
            optimizer = optim.RMSprop(model.parameters(), lr=0.001)

            dataset = Dataset(X[train_index],Y[train_index])
            dataloader = DataLoader(dataset, shuffle=True, batch_size=hparams.per_batch, num_workers=0)

            for epoch in range(hparams.epoch):
                start = time.time()
                running_loss = 0.0
                running_correct = 0 
                per_batch = hparams.per_batch
                batch_size = math.ceil(X[train_index].shape[0] / per_batch)

                for data in dataloader:
                    # load data
                    input, label = data
                    optimizer.zero_grad()
                    bce, sparse_loss = model(input)
                    y_pred = bce > 0.5
                    loss = criterion(bce, label)
                    loss += sparse_loss
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    running_correct += torch.sum(y_pred == label) 

                    '''after batch is done'''

                '''after epoch is done'''
                train_loss = running_loss/len(dataset)
                train_acc = (running_correct/len(dataset)) * 100

                print('subject%d, fold%d' % (i+1, f+1))
                print('epoch%d, train_loss: %.3f' % (epoch+1, train_loss))
                print('epoch%d, train_accuracy: %.3f' % (epoch+1, train_acc))

                end = time.time()
                print('epoch time : ',end-start)

                total_train_loss[i, f, epoch] = train_loss
                total_train_acc[i, f, epoch] = train_acc

                # for validation
                with torch.no_grad():
                    val_dataset = Dataset(X[test_index],Y[test_index])
                    val_dataloader = DataLoader(val_dataset, batch_size=hparams.per_batch, num_workers=0)
                    corr = 0
                    loss = 0
                    for val_data in val_dataloader:
                        val_input, val_label = val_data
                        bce, sparse_loss = model(val_input)
                        y_pred = bce > 0.5
                        loss += criterion(bce, val_label)
                        loss += sparse_loss
                        corr += torch.sum(y_pred == val_label).item()
                    loss /= len(val_dataset)
                    total_val_loss[i, f, epoch] = loss
                    acc = (corr/len(val_dataset)) * 100
                    total_val_acc[i, f, epoch] = acc
                    print('------------------------epoch%d test test_loss: %.3f' % (epoch+1,loss))
                    print('------------------------epoch%d test test_accuracy: %.3f' % (epoch+1, acc))

            '''after fold is done'''

        '''after subject is done'''

        # plt.plot(train_loss)
        # plt.plot(val_loss)
        # plt.legend(['train_loss', 'test_loss'], loc="upper right")

    '''finish'''
    print('train finished')
    print('train_loss : ', torch.mean(total_train_loss, (1,2)))
    print('train_acc : ', torch.mean(total_train_acc, (1,2)))
    print('val_loss : ', torch.mean(total_val_loss, (1,2)))
    print('val_acc : ', torch.mean(total_val_acc, (1,2)))

    total_train_loss_cpu = total_train_loss.to('cpu')
    total_train_acc_cpu = total_train_acc.to('cpu')
    total_val_loss_cpu = total_val_loss.to('cpu')
    total_val_acc_cpu = total_val_acc.to('cpu')

    with open('train_loss_9.pkl', 'wb') as f:
        pickle.dump(total_train_loss_cpu, f)
    with open('train_acc_9.pkl', 'wb') as f:
        pickle.dump(total_train_acc_cpu, f)
    with open('val_loss_9.pkl', 'wb') as f:
        pickle.dump(total_val_loss_cpu, f)
    with open('val_acc_9.pkl', 'wb') as f:
        pickle.dump(total_val_acc_cpu, f)
    # plt.show()
