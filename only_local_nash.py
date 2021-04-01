import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.stats import ortho_group
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from torch.optim.optimizer import required
matplotlib.style.use('ggplot')
plt.rcParams['axes.facecolor'] = '#f9f9f9'
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams["figure.figsize"] = (8, 8)
np.random.seed(11)

lambda_ = 100


class GDA(optim.Optimizer):
    def __init__(self, params, lr=required, momentum=0, alpha=1.0, eta=0.1):
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eta=eta)
        super(GDA, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            alpha = group['alpha']

            p_0 = group['params'][0]
            p_1 = group['params'][1]

            if p_0.grad is None or p_1.grad is None:
                continue
            grad_0, d_p_0 = p_0.tmp
            grad_1, d_p_1 = p_1.tmp

            param_state_0 = self.state[p_0]
            param_state_1 = self.state[p_1]
            if 'previous_iterate' not in param_state_0:
                # - 9.0888
                prev_p_0 = param_state_0['previous_iterate'] = p_0.data.detach() - 1
                prev_p_1 = param_state_1['previous_iterate'] = p_1.data.detach() + 1
                # + 5.6150
            else:
                prev_p_0 = param_state_0['previous_iterate']
                prev_p_1 = param_state_1['previous_iterate']

            param_state_0['previous_iterate'] = p_0.data
            param_state_1['previous_iterate'] = p_1.data

            # import ipdb; ipdb.set_trace()
            temp = alpha * p_0.data + momentum * (p_0.data - prev_p_0.data) + group['eta'] * d_p_0 * (p_1.data - prev_p_1.data) + group['lr'] * grad_0
            p_1.data = alpha * p_1.data + momentum * (p_1.data - prev_p_1.data) - group['eta'] * d_p_1 * (p_0.data - prev_p_0.data)- group['lr'] * grad_1
            p_0.data = temp

            # p_0.data = p_0.data + momentum * (p_0.data - prev_p_0.data) + group['lr'] * d_p_0
            # p_1.data = p_1.data + momentum * (p_1.data - prev_p_1.data) - group['lr'] * d_p_1

        return loss

def plot_surface():
    points = []
    X = np.linspace(-20, 20, 100)
    Y = np.linspace(-20, 20, 100)
    for x in X:
        for y in Y:
            points += [[x, y]]
    points = Variable(torch.FloatTensor(np.array(points)))
    surf = net(points)
    surf = surf.data.numpy().reshape((100, 100)).T
    X, Y = np.meshgrid(X, Y)
    plt.contourf(X, Y, surf, 18, alpha=.5, cmap='RdBu_r')

def ModifiedGDA(net, x_init, y_init, h, k, rho, q, lr):
    net.x.data = torch.FloatTensor([x_init])
    net.y.data = torch.FloatTensor([y_init])
    xys = []
    # h = 1e-1
    # k = 0.3
    # rho = 5
    # q = 0.001

    alpha = (1 - k * h ** 2 + rho * h) / (1 + rho * h)
    beta = 1 / (1 + rho * h)
    eta = (q * h) / (1 + rho * h)
    # lr = 0.005

    opt = GDA(net.parameters(), lr=lr, momentum=beta, alpha=alpha, eta=eta)
    for i in range(1000):
        xys += [[net.x.data[0] + 0, net.y.data[0] + 0]]
        loss = net()
        opt.zero_grad()
        loss.backward(create_graph=True)
        snd_d = torch.autograd.grad(net.x.grad, net.y, retain_graph=True, create_graph=True)[0]
        net.x.tmp = (net.x.grad.data, snd_d.data)
        net.y.tmp = (net.y.grad.data, snd_d.data)

        opt.step()
    xys = np.array(xys)
    plt.plot(xys[:, 0], xys[:, 1], lw=2, color='r', label='Ours')
    plt.savefig('new_ode_res/h: ' + str(h) + ' k: ' + str(k) +
                ' rho: ' + str(rho) + ' q: ' + str(q) +
                ' lr: ' + str(lr) +
                '.png')
    plt.close()


def SimGD(net, x_init, y_init):
    net.x.data = torch.FloatTensor([x_init])
    net.y.data = torch.FloatTensor([y_init])
    xys = []
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.0)
    for i in range(1000):
        xys += [[net.x.data[0] + 0, net.y.data[0] + 0]]
        loss = net()
        opt.zero_grad()
        loss.backward(create_graph=True)
        net.x.grad.data = -net.x.grad.data
        opt.step()
    xys = np.array(xys)
    plt.plot(xys[:, 0], xys[:, 1], lw=2, color='#01117C', label='SimGD')


def ODE_naive(net, x_init, y_init):
    net.x.data = torch.FloatTensor([x_init])
    net.y.data = torch.FloatTensor([y_init])
    xys = []
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.0)
    for i in range(1000):
        xys += [[net.x.data[0] + 0, net.y.data[0] + 0]]
        loss = net()
        opt.zero_grad()
        loss.backward(create_graph=True)
        net.x.grad.data = -net.x.grad.data
        j_row_1 = torch.autograd.grad(net.x.grad, net.parameters(),
                                      retain_graph=True, create_graph=True)
        j_row_2 = torch.autograd.grad(net.y.grad, net.parameters(),
                                      retain_graph=True, create_graph=True)
        joc = torch.cat(
            (torch.cat(j_row_1).unsqueeze(0), torch.cat(j_row_2).unsqueeze(0)))
        omega = torch.cat((net.x.grad.unsqueeze(0), net.y.grad.unsqueeze(0)))
        j_t_j = torch.mm(joc.transpose(0, 1), joc)
        j_lambda = j_t_j + lambda_ * Variable(torch.eye(2))
        omega = 0.5 * (omega + torch.mm(joc.transpose(0, 1),
                       torch.mm(torch.inverse(j_lambda),
                       torch.mm(joc.transpose(0, 1), omega))))
        # omega = 0.5 * (omega + torch.mm(torch.mm(joc.transpose(0, 1), torch.inverse(joc)), omega))

        net.x.grad.data = omega[0].data
        net.y.grad.data = omega[1].data
        opt.step()
    xys = np.array(xys)
    plt.plot(xys[:, 0], xys[:, 1], lw=2, color='g', label="LSS_ODE")


def LSS(net, x_init, y_init):
    net.x.data = torch.FloatTensor([x_init])
    net.y.data = torch.FloatTensor([y_init])
    xys = []
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.0)
    a_n = 0.004
    b_n = 0.005
    kesi_1 = 1e-4
    kesi_2 = 1e-4
    for i in range(1000):
        # a_n = 0.001 / ((i + 1) ** 0.6)
        # b_n = 0.001 / ((i + 1) ** 0.59)
        xys += [[net.x.data[0] + 0, net.y.data[0] + 0]]
        loss = net()
        opt.zero_grad()
        loss.backward(create_graph=True)
        # net.x.grad.data = -net.x.grad.data
        omega = torch.cat((-net.x.grad.unsqueeze(0), net.y.grad.unsqueeze(0)))
        f_lambda = kesi_1 * (1 - torch.exp(- torch.norm(omega, p=2) ** 2))

        jtv_row_1 = torch.autograd.grad(
            torch.mm(omega.transpose(0, 1), net.v), net.x, retain_graph=True,
            create_graph=True)[0]
        jtv_row_2 = torch.autograd.grad(
            torch.mm(omega.transpose(0, 1), net.v), net.y, retain_graph=True,
            create_graph=True)[0]
        jtv = torch.cat((jtv_row_1.unsqueeze(1), jtv_row_2.unsqueeze(1)))
        jv = torch.autograd.grad(
            torch.mm(jtv.transpose(0, 1), net.v), net.v, retain_graph=True,
            create_graph=True)[0]

        g_v = torch.autograd.grad(
            torch.norm(jv - omega, p=2) +
            f_lambda * torch.norm(net.v, p=2), net.v, retain_graph=True)[0]

        net.v.data = net.v.data - b_n * g_v.data

        jtv_norm_2 = torch.norm(jtv, p=2)
        net.x.data = net.x.data - a_n * (
            omega[0] + torch.exp(-kesi_2 * jtv_norm_2) * jtv[0])
        net.y.data = net.y.data - a_n * (
            omega[1] + torch.exp(-kesi_2 * jtv_norm_2) * jtv[1])
        # opt.step()
    xys = np.array(xys)
    plt.plot(xys[:, 0], xys[:, 1], lw=2, color='red', label='LSS')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.x = torch.nn.Parameter(torch.FloatTensor([0]))
        self.y = torch.nn.Parameter(torch.FloatTensor([0]))
        self.v = Variable(torch.FloatTensor([[0], [0]]))
        self.v.requires_grad = True

    def forward(self, xy=None):
        if xy is not None:
            x = xy[:, 0]
            y = xy[:, 1]
            return (torch.exp(-0.01 * (x ** 2 + y ** 2)) *
                    ((0.3 * x ** 2 + y) ** 2 +
                     (0.5 * y ** 2 + x) ** 2))
        return (torch.exp(-0.01 * (self.x ** 2 + self.y ** 2)) *
                ((0.3 * self.x ** 2 + self.y) ** 2 +
                 (0.5 * self.y ** 2 + self.x) ** 2))

net = Net()

# SimGD(net, 10, -10)
# ODE_naive(net, -5, -10)
# ODE_naive(net, 10, -10)
# ModifiedGDA(net, -5, -10)
# ModifiedGDA(net, 10, -10)
# LSS(net, -5, -10)
# LSS(net, 10, -10)

h_list = [1e-1, 5e-1, 10e-1, 1e-2]
k_list = [0.1, 0.3, 0.6, 0.9]
rho_list = [1, 2, 5, 7, 10]
q_list = [0.001, -0.001, 0.003, -0.003, 0.01, -0.01, 0.0001, -0.0001]
lr_list = [0.1, 0.01, 0.001, 0.0001]


# h_list = [1e-1]
# k_list = [0.1]
# rho_list = [5]
# q_list = [0.001]
# lr_list = [0.1, 0.01]
for h in h_list:
    for k in k_list:
        for rho in rho_list:
            for q in q_list:
                for lr in lr_list:
                    plot_surface()
                    SimGD(net, -5, -10)
                    ModifiedGDA(net, -5, -10, h, k, rho, q, lr)
                # plt.legend()


# plt.show()
