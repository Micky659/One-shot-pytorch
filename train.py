import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from dataset import Trainset
from neuralNet.net import Siamese
from neuralNet.loss import ContrastiveLoss

class Trainer():

    def __init__(self, network, data_path):

        self.NeuralNet = network
        self.Loss = ContrastiveLoss()
        self.optimizer = optim.Adam(self.NeuralNet.parameters(), lr=0.0005)

        if torch.cuda.is_available():
            self.NeuralNet = self.NeuralNet.cuda()
        if torch.cuda.device_count() > 1:
            self.NeuralNet = nn.DataParallel(self.NeuralNet)

        self.dataset = Trainset(data_path)

    @staticmethod
    def progress_bar(curr_epoch, curr_batch, batch_num, loss):
        percent = curr_batch / batch_num
        last = int((percent * 1000) % 10)
        percent = round(percent * 100)
        bar = 'Epoch: {:3d} '.format(curr_epoch)
        bar += 'Batch: {:3d} '.format(curr_batch)
        bar += 'Loss: {:.4f} '.format(loss)
        bar += '|' + '#' * int(percent)
        if curr_batch != batch_num:
            bar += '{}'.format(last)
            bar += ' ' * (100 - int(percent)) + '|'
            print('\r' + bar, end='')
        else:
            bar += '#'
            bar += ' ' * (100 - int(percent)) + '|'
            print('\r' + bar)

    def save_state_dict(self, dict_path):
        torch.save(self.NeuralNet.state_dict(), dict_path)

    def train(self, batch_size, epochs):
        data_loader = DataLoader(self.dataset,
                                   shuffle=True,
                                   num_workers=4,
                                   batch_size=batch_size)

        counter = []
        loss_history = []
        iteration_number = 0
        batch_num = self.dataset.__len__() // batch_size

        for epoch in range(0, epochs):
            for i, data in enumerate(data_loader, 0):
                img0, img1, label = data
                if torch.cuda.is_available():
                    img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                self.optimizer.zero_grad()
                output1, output2 = self.NeuralNet(img0, img1)
                loss = self.Loss(output1, output2, label)
                loss.backward()
                self.optimizer.step()
                self.progress_bar(epoch + 1, i, batch_num, loss.item())
            iteration_number += 1
            counter.append(iteration_number)
            loss_history.append(loss.item())

if __name__ == '__main__':
    net = Siamese()
    train_path = 'D:/Projects/One-shot-pytorch/trainset'
    trainer = Trainer(net, train_path)
    trainer.train(batch_size=16, epochs=50)
    trainer.save_state_dict('model/checkpoint')