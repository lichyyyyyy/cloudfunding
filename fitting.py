import pickle 
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import imageio
import random
import torch.nn as nn


BATCH_SIZE = 64
EPOCH = 200
TRAIN_DATASET = 0.9

def basic(x, a, b, c, d):
    a,b,c,d = convert2numpy([a,b,c,d])
    return np.add(
        a,
        np.divide(
            b,
            np.add(
                c,
                np.exp(
                    np.multiply(
                        np.negative(d),
                        x
                    )
                )
            )
        )
    )


def four_pl(x, a, b, c, d):
    return np.add(
        c,
        np.divide(
            np.subtract(
                d, c
            ),
            np.add(
                1,
                np.exp(
                    np.multiply(
                        a,
                        np.subtract(
                            x,
                            b
                        )
                    )
                )
            )
        )
    )


def five_pl(x, a, b, c, d, g):
    return np.add(
        c,
        np.divide(
            np.subtract(
                1 + d,
                c
            ),
            np.power(
                np.add(
                    1,
                    np.exp(
                        np.multiply(
                            -a,
                            np.subtract(
                                x,
                                b
                            )
                        )
                    )
                ),
                g
            )
        )
    )


def five(x, a, b, c, d, g):
    return np.add(
        d,
        np.divide(
            np.subtract(
                1 + a,
                d
            ),
            np.power(
                np.add(
                    1,
                    np.power(
                        np.divide(
                            x,
                            c
                        ),
                        b
                    )
                ),
                g
            )
        )
    )


def bass(x, m, p, q):
    m,p,q = convert2numpy([m,p,q])
    print(m,p,q)
    return np.multiply(
        m,
        np.divide(
            q - np.multiply(
                p,
                np.exp(
                    np.negative(np.multiply(
                        -(p + q),
                        x
                    ))
                )),
            q + np.multiply(
                q,
                np.exp(
                    np.negative(np.multiply(
                        -(p + q),
                        x
                    ))
                )
            )
        )
    )


def convert2numpy(params):
    return [param.detach().numpy() for param in params]


class Net(torch.nn.Module):
    def __init__(self, input_channel, output_channel, n_features, n_output, batch):
        super(Net, self).__init__()
        self.conv = torch.nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)  
        self.pool = torch.nn.MaxPool1d(kernel_size=2, padding=0)
        self.fc = torch.nn.Linear(int(n_features*batch*output_channel/2), 516)  
        self.fc1 = torch.nn.Linear(516, 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, n_output)

    def forward(self, x, n_features, output_channel, batch, input_channel):
        if(batch < BATCH_SIZE):
            target = torch.zeros(BATCH_SIZE, input_channel, n_features)
            target[:batch, :, :] = x
            x = target
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(-1, int(n_features*BATCH_SIZE*output_channel/2))
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x[0][0] = 5 if x[0][0] > 30 or x[0][0] < -30 else x[0][0]
        x[0][1] = 5 if x[0][1] > 30 or x[0][1] < -30 else x[0][1]
        x[0][2] = 5 if x[0][2] > 30 or x[0][2] < -30 else x[0][2]
        x[0][3] = 5 if x[0][3] > 30 or x[0][3] < -30 else x[0][3]
        return x


def loss_func(model, params, pairs):
    params = params[0]
    prediction, money = Variable(torch.DoubleTensor([]), requires_grad=True), torch.DoubleTensor([])
    if model == 2:
        for pair in pairs:
            days = np.array([item[0] for item in pair])
            prediction = torch.cat((prediction, torch.from_numpy(basic(days, params[0], params[1], params[2], params[3]))), 0)
            money = torch.cat((money, torch.from_numpy(np.array([item[1] for item in pair])).double()), 0)

        loss = nn.MSELoss()
        return loss(prediction, money)


torch.manual_seed(1)    

x = pickle.load(open('features.p', 'rb'))  # x data (tensor), shape=(5958, 28)
y = pickle.load(open('dataset.p', 'rb'))   # y data (tensor), shape=(5958, n, 2)
x = Variable(torch.from_numpy(np.array(x))).unsqueeze(1).double()     # shape=(5958, 1, 28)

net = Net(input_channel=1, output_channel=12, n_features=28, n_output=4, batch=BATCH_SIZE)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

train_dataset = Data.TensorDataset(x[:int(TRAIN_DATASET*5958)])
test_dataset = Data.TensorDataset(x[int(TRAIN_DATASET*5958):])

my_images = []
fig, ax = plt.subplots(figsize=(16,10))

# start training
for epoch in range(EPOCH):
    loader = iter(Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers = 0, shuffle=True))
    for step, b_x in enumerate(loader):
        batch = len(b_x[0])
        b_x = Variable(b_x[0]).float()    # shape=(batch_size, 1,  28)
        b_y = np.array(y[step*BATCH_SIZE : step*BATCH_SIZE+batch])

        params = net(b_x, output_channel=12, n_features=28, batch=batch, input_channel=1)     
        loss = loss_func(2, params, b_y)     

        optimizer.zero_grad()   
        loss.backward()         
        optimizer.step()       
        if step == 1:
            print('epoch {}: loss: {}; params: {}'.format(epoch, loss.data.numpy(), params.data.numpy()))


