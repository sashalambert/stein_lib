"""
https://github.com/alisiahkoohi/Langevin-dynamics
Ali Siahkoohi
"""


import torch
import numpy as np
from .precondSGLD import pSGLD
from tqdm import tqdm
import copy

class LangevinDynamics(object):
    def __init__(
            self,
            lr=1e-2,
            lr_final=1e-4,
            max_itr=None,
            gamma=-0.55
    ):
        super(LangevinDynamics, self).__init__()

        self.lr = lr
        self.lr_final = lr_final
        self.gamma = gamma
        self.counter = 0.0
        self.optim = None
        self.lr_fn = None
        self.max_itr = max_itr

    def sample(self, x, model):
        self.lr_decay()
        self.optim.zero_grad()
        loss = -1. * model.log_prob(x).sum()
        loss.backward()
        self.optim.step()
        self.counter += 1
        return copy.deepcopy(x.data), loss.item()

    def decay_fn(self, lr=1e-2, lr_final=1e-4, max_itr=1e4, gamma=-0.55):
        b = max_itr/((lr_final/lr)**(1/gamma) - 1.0)
        a = lr/(b**gamma)
        def lr_fn(t, a=a, b=b, gamma=gamma):
            return a*((b + t)**gamma)
        return lr_fn

    def lr_decay(self):
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_fn(self.counter)

    def apply(self, x, model):
        hist_samples = []
        loss_log = []
        x.requires_grad = True
        self.optim = pSGLD([x], self.lr, weight_decay=0.0)
        self.lr_fn = self.decay_fn(
            lr=self.lr,
            lr_final=self.lr_final,
            max_itr=self.max_itr,
            gamma=self.gamma,
        )
        for j in tqdm(range(self.max_itr)):
            est, loss = self.sample(x, model)
            loss_log.append(loss)
            # if j % 10 == 0:
            hist_samples.append(est.cpu().numpy())
        particles = est
        p_hist = np.array(hist_samples)
        return (particles, p_hist)


class MetropolisAdjustedLangevin(object):
    def __init__(
            self,
            x,
            func,
            lr=1e-2,
            lr_final=1e-4,
            max_itr=1e4,
    ):
        super(MetropolisAdjustedLangevin, self).__init__()

        self.x = [
            torch.zeros(x.shape, device=x.device, requires_grad=True),
            torch.zeros(x.shape, device=x.device, requires_grad=True)
            ]
        self.x[0].data = x.data
        self.x[1].data = x.data

        self.loss = [torch.zeros([1], device=x.device),
                     torch.zeros([1], device=x.device)]
        # self.loss[0] = func(self.x[0])
        self.loss[0] = -1. * func.log_prob(self.x[0]).sum()
        self.loss[1].data = self.loss[0].data

        self.grad = [torch.zeros(x.shape, device=x.device),
                     torch.zeros(x.shape, device=x.device)]
        self.grad[0].data = torch.autograd.grad(self.loss[0], [self.x[0]],
            create_graph=False)[0].data
        self.grad[1].data = self.grad[0].data

        self.optim = pSGLD([self.x[1]], lr, weight_decay=0.0)
        self.lr = lr
        self.lr_final = lr_final
        self.max_itr = max_itr
        self.func = func
        self.lr_fn = self.decay_fn(lr=lr, lr_final=lr_final, max_itr=max_itr)
        self.counter = 0.0

    def sample(self):
        accepted = False
        self.lr_decay()
        while not accepted:
            self.x[1].grad = self.grad[1].data
            self.P = self.optim.step()
            # self.loss[1] = self.func(self.x[1])
            self.loss[1] = -1. * self.func.log_prob(self.x[1]).sum()
            self.grad[1].data = torch.autograd.grad(
                self.loss[1], [self.x[1]], create_graph=False)[0].data

            alpha = min([1.0, self.sample_prob()])
            if torch.rand([1]) <= alpha:
                self.grad[0].data = self.grad[1].data
                self.loss[0].data = self.loss[1].data
                self.x[0].data = self.x[1].data
                accepted = True
            else:
                self.x[1].data = self.x[0].data
        self.counter += 1
        return copy.deepcopy(self.x[1].data), self.loss[1].item()

    def proposal_dist(self, idx):
        return (-(.25 / self.lr_fn(self.counter)) *
                torch.norm(self.x[idx] - self.x[idx^1] -
                           self.lr_fn(self.counter)*self.grad[idx^1]/self.P)**2
        )

    def sample_prob(self):
        return torch.exp(-self.loss[1] + self.loss[0]) * \
            torch.exp(self.proposal_dist(0) - self.proposal_dist(1))

    def decay_fn(self, lr=1e-2, lr_final=1e-4, max_itr=1e4):
        gamma = -0.55
        b = max_itr/((lr_final/lr)**(1/gamma) - 1.0)
        a = lr/(b**gamma)
        def lr_fn(t, a=a, b=b, gamma=gamma):
            return a*((b + t)**gamma)
        return lr_fn

    def lr_decay(self):
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_fn(self.counter)

    def apply(self):
        hist_samples = []
        loss_log = []
        for j in tqdm(range(int(self.max_itr))):
            est, loss = self.sample()
            loss_log.append(loss)
            # if j % 10 == 0:
            hist_samples.append(est.cpu().numpy())
        particles = est
        p_hist = np.array(hist_samples)
        return (particles, p_hist)