import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from torch.utils.data.dataset import Dataset
import math

from sklearn.model_selection import train_test_split


class AutoEncoderDataset(Dataset):
    """Dataset wrapper for autoencoding """

    def __init__(self, data=None):
        self.data = data

    def __getitem__(self, index):
        return self.data[index], self.data[index]

    def __len__(self):
        return len(self.data)

def reduceData(data, output_dim):

    # zero mean
    data = data - np.mean(data,axis=0)
    # return Variable(torch.FloatTensor(data))

    print('data')
    print(data)

    # shuffle data and split it
    # np.random.shuffle(data)

    training, development = train_test_split(data, test_size=0.2)
    training = data
    print('training')
    print(training)

    print('development')
    print(development)

    # converting train data into torch


    #shuffle train_data

    train_data = torch.FloatTensor(training)

    print('train_data')
    print(train_data)

    #converting development data into torch
    dev_data = torch.FloatTensor(development)

    print('dev_data')
    print(dev_data)

    # Hyper Parameters
    EPOCH = 10
    BATCH_SIZE = 32
    LR = 0.01

    input_dim = train_data.size()[1]

    print('input_dim')
    print(input_dim)
    dataset = AutoEncoderDataset(train_data)
    dev_dataset = AutoEncoderDataset(train_data)
    print('dataset')
    print(dataset)
    print('dev_dataset')
    print(dev_dataset)

    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    # Dev data loader
    dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

    autoencoder = AutoEncoder(input_dim, output_dim)
    autoencoder = autoencoder.cuda()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss(size_average=False)
    bestLoss = float("inf")

    iterationList = []

    trainList = []

    devList = []

    # autoencoder = autoencoder.cuda()
    for epoch in range(EPOCH):
        num_of_train_samples = 0
        loss_train = 0
        for step, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()  # clear gradients for this training step
            b_x = Variable(x)

            b_x = b_x.cuda()
            # print('b_x')
            # print(b_x)

            # b_x = b_x.cuda()

            encoded, decoded = autoencoder(b_x)
            # print('encoded')
            # print(encoded)
            # print('decoded')
            # print(decoded)
            loss = loss_func(decoded, b_x)  # mean square error
            # print('loss')
            # print(loss)
            loss.backward()  # backpropagation, compute gradients
            # for p in autoencoder.parameters():
            #     print('p')
            #     print(p)
            #     print('p.grad')
            #     print(p.grad)
            optimizer.step()  # apply gradients

            loss_train += loss.data.cpu().numpy()[0]
            num_of_train_samples += b_x.size()[0]

        avg_train_loss = loss_train / float(num_of_train_samples)
        print('Epoch: ', epoch, '| train loss: %.8f' % avg_train_loss)
        iterationList.append(epoch)
        trainList.append(avg_train_loss)

        lossDev = 0
        num_of_dev_samples = 0

        for step, (d_x, _) in enumerate(dev_loader):
            dev_x = Variable(d_x)
            dev_x = dev_x.cuda()
            num_of_dev_samples += dev_x.size()[0]
            encodedDev, decodedDev = autoencoder(dev_x)

            # print('d_x')
            # print(d_x)
            # print('encodedDev')
            # print(encodedDev)
            # print('decodedDev')
            # print(decodedDev)

            lossDev += loss_func(decodedDev, dev_x).cpu().data.numpy()[0]
        avg_dev_loss = lossDev / float(num_of_dev_samples)
        print('Epoch: ', epoch, '| dev loss: %.8f' % avg_dev_loss)
        devList.append(avg_dev_loss)

        if avg_dev_loss < bestLoss:
            print('New Best Dev Loss %s' % avg_dev_loss)
            print('bestLoss %s' %bestLoss)
            autoencoder = autoencoder.cuda()
            print('Saving model')

            torch.save(autoencoder, "model.torch")
            bestLoss = avg_dev_loss

    autoencoder = torch.load("model.torch")

    plt.plot(iterationList, trainList, color='g')
    plt.plot(iterationList, devList, color='orange')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title('Iterations vs Loss')
    plt.savefig('LossVsIterationsnewAloiNL.png')

    # get encoded and decoded data
    view_data = Variable(torch.FloatTensor(data))
    view_data = view_data.cuda()
    encoded_data, decoded_data = autoencoder(view_data)

    #print(decoded_data.shape)
    #return decoded_datahpe
    print (encoded_data.shape)
    return encoded_data


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AutoEncoder, self).__init__()
        # d_prime_1 is longer one
        # d_prime = (input_dim+output_dim)/2
        d_prime = input_dim
        self.encoder = nn.Sequential(

            # 1 One Hidden Layer
            # nn.Linear(input_dim, output_dim),
            # nn.Tanh(),

            # 2 Two Hidden Layers
            # d_prime_1 = (input_dim+output_dim)/2
            # d_prime_2 = input_dim

            # nn.Linear(input_dim, d_prime),
            # nn.Tanh(),
            # nn.Linear(d_prime, output_dim),
            # nn.Tanh(),
            # 3 Two Hidden Layers

            # nn.Linear(input_dim, output_dim),
            nn.Linear(input_dim, d_prime),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(d_prime, output_dim),
            # nn.Tanh(),

        )
        self.decoder = nn.Sequential(

            # 1 One Hidden Layer
            # nn.Linear(output_dim, input_dim),
            # nn.Tanh(),


            # 2 Two Hidden Layers
            # d_prime_1 = (input_dim+output_dim)/2
            # d_prime_2 = input_dim

            # nn.Linear(output_dim, input_dim),
            # nn.Linear(output_dim, output_dim),
            # nn.Tanh(),
            # nn.ReLU(),
            #nn.Linear(output_dim, input_dim),
            nn.Linear(output_dim, d_prime),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(d_prime, input_dim),


            # nn.Tanh(),
            # 3 Two Hidden Layers

            # nn.Linear(input_dim, d_prime)
            # nn.Tanh()
            # nn.Linear(d_prime, output_dim)
            # nn.Tanh()

        )

    def forward(self, x):
        encoded = self.encoder(x)
        # print (type(encoded))
        # numpy_encoded = encoded.cpu().data.numpy()
        # norm_encoded = numpy_encoded / np.linalg.norm(numpy_encoded, axis=1, keepdims=True)
        # torch_encoded = torch.from_numpy(norm_encoded)
        # print (type(torch_encoded))
        # var_encoded = Variable(torch_encoded)
        # var_encoded = var_encoded.cuda()
        # decoded = self.decoder(var_encoded)
        decoded = self.decoder(encoded)
        return encoded, decoded

# encoded_data = reduceData(np.random.random_sample((1000,128)), 3)
# return reduceData(dataset)
# return encoder_data
# print(type(encoded_data))
# encoded_data =encoded_data.data.numpy()
# print(encoded_data.shape)
# print(encoded_data)
