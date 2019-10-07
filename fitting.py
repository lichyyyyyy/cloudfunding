import pickle 
import torch
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import imageio
import random
import torch.nn as nn
import torch.nn.functional as F


BATCH_SIZE = 64
EPOCH = 200
TRAIN_DATASET = 0.9

def basic(x, a, b, c, d):
    a,b,c,d = param_constraints([a,b,c,d])
    a = a.squeeze()
    b = b.squeeze()
    c = c.squeeze()
    d = d.squeeze()
    return torch.add(
        a,
        torch.div(
            b,
            torch.add(
                c,
                torch.exp(
                    torch.mul(
                        torch.neg(d),
                        x
                    )
                )
            )
        )
    )


def four_pl(x, a, b, c, d):
    a,b,c,d = param_constraints([a,b,c,d])
    return torch.add(
        c,
        torch.div(
            torch.add(
                d, 
                torch.neg(c)
            ),
            torch.add(
                1,
                torch.exp(
                    torch.mul(
                        a,
                        torch.add(
                            x,
                            torch.neg(b)
                        )
                    )
                )
            )
        )
    )


def five_pl(x, a, b, c, d, g):
    a,b,c,d,g = param_constraints([a,b,c,d,g])
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


def param_constraints(params):
    highest = torch.tensor(10).float()
    lowest = torch.tensor(-10).float()
    for index, param in enumerate(params):
        param = highest if torch.equal(torch.gt(param, highest), torch.tensor(True)) else param
        param = lowest if torch.equal(torch.lt(param, lowest), torch.tensor(True)) else param
    return params


class Net(torch.nn.Module):
    def __init__(self, n_features, n_output):
        super(Net, self).__init__()
        self.fc = torch.nn.Linear(n_features, n_features)  
        self.fc1 = torch.nn.Linear(n_features, 32)
        self.fc2 = torch.nn.Linear(n_features, n_output)
        self.lr = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.lr(x)
        x = self.fc(x)
        x = self.lr(x)
        x = self.fc2(x)
        '''x[0][0] = 5 if x[0][0] > 30 or x[0][0] < -30 else x[0][0]
        x[0][1] = 5 if x[0][1] > 30 or x[0][1] < -30 else x[0][1]
        x[0][2] = 5 if x[0][2] > 30 or x[0][2] < -30 else x[0][2]
        x[0][3] = 5 if x[0][3] > 30 or x[0][3] < -30 else x[0][3]'''
        #print(x)
        return x


def loss_func(model, params, pairs):
    prediction, money = Variable(torch.FloatTensor([]), requires_grad=True), torch.FloatTensor([])
    mse = Variable(torch.FloatTensor([]), requires_grad=True)
    loss = nn.MSELoss()
    if model == 2:
        for index in range(len(params)):
            pair = pairs[index]
            param = params.narrow(0, index, 1)
            days = torch.tensor(np.array([item[0] for item in pair]), requires_grad=True, dtype=torch.float64)
            prediction = torch.cat((prediction, basic(days, param.narrow(1, 0, 1), param.narrow(1, 1, 1), 
                param.narrow(1, 2, 1), param.narrow(1, 3, 1)).float()), 0)
            money = torch.cat((money, torch.from_numpy(np.array([item[1] for item in pair])).float()), 0)
            mse = torch.cat((mse, torch.FloatTensor([loss(prediction, money).float()])), 0)

        return torch.mean(mse)


torch.manual_seed(1)    

x = pickle.load(open('features.p', 'rb'))  # x data (tensor), shape=(5958, 28)
y = pickle.load(open('dataset.p', 'rb'))   # y data (tensor), shape=(5958, n, 2)
x = Variable(torch.from_numpy(np.array(x))).float()     # shape=(5958, 28)

net = Net(n_features=28, n_output=4)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

train_dataset = Data.TensorDataset(x[:int(TRAIN_DATASET*5958)])
test_dataset = Data.TensorDataset(x[int(TRAIN_DATASET*5958):])

my_images = []
fig, ax = plt.subplots(figsize=(16,10))

# start training
for epoch in range(EPOCH):
    loader = iter(Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers = 0))   #, shuffle=True
    for step, b_x in enumerate(loader):
        batch = len(b_x[0])
        b_x = Variable(b_x[0], requires_grad=True)    # shape=(batch_size, 28)
        b_y = np.array(y[step*BATCH_SIZE : step*BATCH_SIZE+batch])

        params = net(b_x)
        params.retain_grad()
        loss = loss_func(2, params, b_y)
        loss.retain_grad()
        optimizer.zero_grad()
        loss.backward()
        #print(params.grad)
        optimizer.step()
        if step == 0:
            print('epoch {}: loss: {}'.format(epoch, loss.data.numpy()))


