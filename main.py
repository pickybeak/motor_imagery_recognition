import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import hyperparameters as hparams
import torch.optim as optim
import pickle
import math
import statistics

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

    def forward(self, x):
        B, _, _, _ = x.shape
        sparse, bce = self.recognizer(x)
        sparse_loss = torch.norm(sparse, 1) / B
        return bce, sparse_loss

if __name__ == '__main__':

    # 9 subject, 10 fold, 500 epoch
    total_train_loss, total_train_acc = [], []
    total_val_loss, total_val_acc = [], []

    subject_train_loss, subject_train_acc = [], []
    subject_val_loss, subject_val_acc = [], []
    fold_train_loss, fold_train_acc = [], []
    fold_val_loss, fold_val_acc = [], []
    epoch_train_loss, epoch_train_acc = [], []
    epoch_val_loss, epoch_val_acc = [], []

    for i in range(1, 10):
        file = 'A0%d_64x64_scipy2_cv10.pkl'%i
        with open(file, "rb") as f:
            dataset = pickle.load(f)
            X = dataset["X"]
            Y = dataset["y"]
            Y -=1
            folds = dataset["folds"]

        # plt.subplot(3,3,i)
        model = Model()
        criterion = torch.nn.BCELoss()
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)

        for f, (train_index, test_index) in enumerate(folds):

            epoch_train_loss, epoch_train_acc = [], []
            epoch_val_loss, epoch_val_acc = [], []

            for epoch in range(hparams.epoch):
                running_loss = 0.0
                running_acc = 0.0
                per_batch = hparams.per_batch
                batch_size = math.ceil(X[train_index].shape[0] / per_batch)

                for batch_index in range(batch_size):
                    # load data
                    input = torch.Tensor(X[train_index][batch_index:(per_batch*(batch_index+1))])
                    label = torch.Tensor(Y[train_index][batch_index:(per_batch*(batch_index+1))])
                    optimizer.zero_grad()
                    bce, sparse_loss = model(input)
                    y_pred = bce > 0.5
                    loss = criterion(bce, label)
                    loss += sparse_loss
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    running_acc += (torch.sum(y_pred == label) / y_pred.shape[0]) * 100

                    if batch_index % 1 == 0:
                        pass

                '''after epoch is done'''
                train_loss = running_loss/batch_size
                train_acc = running_acc/batch_size

                print('subject%d, fold%d' % (i, f+1))
                print('epoch%d, train_loss: %.3f' % (epoch+1, train_loss))
                print('epoch%d, train_accuracy: %.3f' % (epoch+1, train_acc))

                epoch_train_loss.append(train_loss)
                epoch_train_acc.append(train_acc)

                # for validation
                with torch.no_grad():
                    input = torch.Tensor(X[test_index])
                    label = torch.Tensor(Y[test_index])
                    bce, sparse_loss = model(input)
                    y_pred = bce > 0.5
                    loss = criterion(bce, label)
                    loss += sparse_loss
                    acc = (torch.sum(y_pred == label) / y_pred.shape[0]) * 100
                    epoch_val_loss.append(loss)
                    epoch_val_acc.append(acc)
                    print('------------------------epoch%d test test_loss: %.3f' % (epoch+1,loss))
                    print('------------------------epoch%d test test_accuracy: %.3f' % (epoch+1, acc))

            '''after fold is done'''
            fold_train_loss.append(epoch_train_loss)
            fold_val_loss.append(epoch_val_loss)
            fold_train_acc.append(epoch_train_acc)
            fold_val_acc.append(epoch_val_acc)

        '''after subject is done'''
        subject_train_loss.append(fold_train_loss)
        subject_val_loss.append(fold_val_loss)
        subject_train_acc.append(fold_train_acc)
        subject_val_acc.append(fold_val_acc)
        # with torch.no_grad():
        #     input = torch.Tensor(X[train_index])
        #     label = torch.Tensor(Y[train_index])
        #     bce, sparse_loss = model(input)
        #     y_pred = bce > 0.5
        #     train_acc = (torch.sum(y_pred == label) / y_pred.shape[0]) * 100
        #     train_acc_by_subject.append(float(train_acc))
        #     print('------------------------total train accuracy: %.3f' % train_acc)
        #
        #     input = torch.Tensor(X[test_index])
        #     label = torch.Tensor(Y[test_index])
        #     bce, sparse_loss = model(input)
        #     y_pred = bce > 0.5
        #     val_acc = (torch.sum(y_pred == label) / y_pred.shape[0]) * 100
        #     val_acc_by_subject.append(float(val_acc))
        #     print('------------------------total test accuracy: %.3f' % val_acc)

        # plt.plot(train_loss)
        # plt.plot(val_loss)
        # plt.legend(['train_loss', 'test_loss'], loc="upper right")
    '''finish'''
    print('train finished')
    print('train_loss : ', [statistics.mean(total_train_loss[i][:][:]) for i in range(10)])
    print('train_acc : ', [statistics.mean(total_train_acc[i][:][:]) for i in range(10)])
    print('val_loss : ', [statistics.mean(total_val_loss[i][:][:]) for i in range(10)])
    print('val_acc : ', [statistics.mean(total_val_acc[i][:][:]) for i in range(10)])

    with open('train_loss.pkl', 'wb') as f:
        pickle.dump(total_train_loss, f)
    with open('train_acc.pkl', 'wb') as f:
        pickle.dump(total_train_acc, f)
    with open('val_loss.pkl', 'wb') as f:
        pickle.dump(total_val_loss, f)
    with open('val_acc.pkl', 'wb') as f:
        pickle.dump(total_val_acc, f)
    # plt.show()