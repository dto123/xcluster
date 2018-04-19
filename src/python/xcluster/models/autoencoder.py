import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
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

    #shuffle data and split it
    data = data - np.mean(data,axis=1)

    np.random.shuffle(data)

    training, development = train_test_split(data, test_size=0.2)

    #converting train data into torch
    train_data = torch.from_numpy(training)
    train_data = train_data.type(torch.FloatTensor)

    #train_data = train_data.cuda()

    points, input_dim = train_data.size()
    print (train_data.size())

    #converting development data into torch
    dev_data = torch.from_numpy(development)
    dev_data = dev_data.type(torch.FloatTensor)

    #dev_data = dev_data.cuda()

    dev_points, dev_input_dim = train_data.size()
    print (dev_data.size())
    # Hyper Parameters
    EPOCH = 1000
    BATCH_SIZE = 64
    #LR = 0.005         # learning rate
    LR = 0.01
    DOWNLOAD_MNIST = False
    N_TEST_IMG = 5

    dataset = AutoEncoderDataset(train_data)
    dev_dataset = AutoEncoderDataset(dev_data)
    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    # Dev data loader
    dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    autoencoder = AutoEncoder(input_dim, output_dim)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    #bestLoss = math.inf
    bestLoss = float("inf")

    autoencoder = autoencoder.cuda()
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x.view(-1, input_dim))   # batch x, shape (batch, 28*28)
            b_y = Variable(x.view(-1, input_dim))   # batch y, shape (batch, 28*28)

            b_x = b_x.cuda()
            b_y = b_y.cuda()
            #b_label = Variable(y)               # batch label

            #autoencoder = autoencoder.cuda()

            encoded, decoded = autoencoder(b_x)

            loss = loss_func(decoded, b_y)      # mean square error
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients

            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.8f' % loss.data[0])

                lossDev = 0
                num_of_dev_samples = 0
                for step, (d_x, d_y) in enumerate(dev_loader):
                    dev_x = Variable(d_x.view(-1, dev_input_dim))
                    dev_x = dev_x.cuda()
                    num_of_dev_samples += dev_x.size()[0]

                    encodedDev, decodedDev = autoencoder(dev_x)
                    lossDev += loss_func(decodedDev, dev_x).cpu().data.numpy()[0]
                avg_dev_loss = lossDev / float(num_of_dev_samples)
                print('Epoch: ', epoch, '| dev loss: %.8f' % avg_dev_loss)
                if lossDev < bestLoss:
                    autoencoder = autoencoder.cuda()
                    torch.save(autoencoder, "model.torch")

    autoencoder = torch.load("model.torch")



    # visualize in 3D plot
    view_data = Variable(train_data.view(-1, input_dim))
    view_data = view_data.cuda()
    encoded_data, _ = autoencoder(view_data)

    print(encoded_data.shape)
    return encoded_data.data
    #return encoded_data




class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AutoEncoder, self).__init__()
        #d_prime_1 is longer one
        #d_prime = (input_dim+output_dim)/2
        d_prime = input_dim
        self.encoder = nn.Sequential(

            #1 One Hidden Layer
            #nn.Linear(input_dim, output_dim),
            #nn.Tanh(),

            #2 Two Hidden Layers
            #d_prime_1 = (input_dim+output_dim)/2
            #d_prime_2 = input_dim

            #nn.Linear(input_dim, d_prime),
            #nn.Tanh(),
            #nn.Linear(d_prime, output_dim),
            #nn.Tanh(),
            #3 Two Hidden Layers

            nn.Linear(input_dim, d_prime),
            nn.Tanh(),
            #nn.ReLU(),
            nn.Linear(d_prime, output_dim),
            #nn.Tanh(),

        )
        self.decoder = nn.Sequential(


            #1 One Hidden Layer
            #nn.Linear(output_dim, input_dim),
            #nn.Tanh(),


            #2 Two Hidden Layers
            #d_prime_1 = (input_dim+output_dim)/2
            #d_prime_2 = input_dim

            nn.Linear(output_dim, d_prime),
            nn.Tanh(),
            #nn.ReLU(),
            nn.Linear(d_prime, input_dim),
            #nn.Tanh(),
            #3 Two Hidden Layers

            #nn.Linear(input_dim, d_prime)
            #nn.Tanh()
            #nn.Linear(d_prime, output_dim)
            #nn.Tanh()

        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

#encoded_data = reduceData(np.random.random_sample((1000,128)), 3)
#return reduceData(dataset)
# return encoder_data
#print(type(encoded_data))
#encoded_data =encoded_data.data.numpy()
#print(encoded_data.shape)
#print(encoded_data)
