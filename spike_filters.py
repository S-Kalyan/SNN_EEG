import torch
import numpy as np
from torch.autograd import Variable

class SpikeFilter(object):

    @staticmethod
    def single_exponential_filter(spike_train, dt, tau):
        """

        :param spike_train: spike_train of one neuron; shape: (n)
        :param dt: time step
        :param tau: synaptic time constant
        :return: filtered spike_train
        """
        r = np.zeros(len(spike_train))
        for i in range(1, len(spike_train)):
            # r[i] = r[i-1]*(1 - dt/tau) + (dt/tau)*np.sum(spike_train[:i])
            r[i] = r[i - 1] * (1 - dt / tau) + (dt / tau) * spike_train[i - 1]

        return r

    @staticmethod
    def single_exponential_filter_exp(spike_train, dt, tau):
        r = np.zeros(len(spike_train))
        for i in range(1, len(spike_train)):
            # r[i] = r[i-1]*(1 - dt/tau) + (dt/tau)*np.sum(spike_train[:i])
            r[i] = r[i - 1] * np.exp(-dt / tau) + (1 / tau) * spike_train[i - 1]

        return r

    @staticmethod
    def double_exponential_filter(spike_train, dt, tau_r, tau_d):
        """

        :param spike_train: spike_train of one neuron; shape: (n)
        :param dt: time step
        :param tau_r: synaptic rise time
        :param tau_d: synaptic decay time
        :return: filtered spike_train
        """
        # tau_r, tau_d = torch.tensor(tau_r).type(torch.FloatTensor), torch.tensor(tau_d).type(torch.FloatTensor)
        # dt = torch.tensor(dt).type(torch.FloatTensor)

        r = Variable(torch.zeros(len(spike_train), ))
        h = torch.zeros(len(spike_train))
        for i in range(1, len(spike_train)):
            # h[i] = h[i-1] *(1 - dt/tau_r) + (dt/(tau_r*tau_d))*spike_train[i-1]
            h[i] = h[i - 1] * (1 - dt / tau_r) + (dt / (tau_r * tau_d)) * spike_train[i - 1]
            r[i] = r[i - 1] * (1 - dt / tau_d) + dt * h[i]

        return r

    @staticmethod
    def double_exponential_filter_np(spike_train, dt, tau_r, tau_d):
        """

        :param spike_train: spike_train of one neuron; shape: (n)
        :param dt: time step
        :param tau_r: synaptic rise time
        :param tau_d: synaptic decay time
        :return: filtered spike_train
        """
        # tau_r, tau_d = torch.tensor(tau_r).type(torch.FloatTensor), torch.tensor(tau_d).type(torch.FloatTensor)
        # dt = torch.tensor(dt).type(torch.FloatTensor)

        r = np.zeros(len(spike_train))
        h = np.zeros(len(spike_train))
        for i in range(1, len(spike_train)):
            # h[i] = h[i-1] *(1 - dt/tau_r) + (dt/(tau_r*tau_d))*spike_train[i-1]
            h[i] = h[i - 1] * (1 - dt / tau_r) + (dt / (tau_r * tau_d)) * spike_train[i - 1]
            r[i] = r[i - 1] * (1 - dt / tau_d) + dt * h[i]

        return r

    @staticmethod
    def double_exponential_filter_multi(spike_train, dt, tau_r, tau_d):
        """

        :param spike_train: spike_train of one neuron; shape: (n)
        :param dt: time step
        :param tau_r: synaptic rise time
        :param tau_d: synaptic decay time
        :return: filtered spike_train
        """
        # tau_r, tau_d = torch.tensor(tau_r).type(torch.FloatTensor), torch.tensor(tau_d).type(torch.FloatTensor)
        # dt = torch.tensor(dt).type(torch.FloatTensor)

        r = Variable(torch.zeros(spike_train.shape[0], spike_train.shape[1]))
        h = torch.zeros(spike_train.shape[0], spike_train.shape[1])
        for i in range(1, spike_train.shape[0]):
            # h[i] = h[i-1] *(1 - dt/tau_r) + (dt/(tau_r*tau_d))*spike_train[i-1]
            h[i, :] = h[i - 1, :] * (1 - dt / tau_r) + (dt / (tau_r * tau_d)) * spike_train[i - 1, :]
            r[i, :] = r[i - 1, :] * (1 - dt / tau_d) + dt * h[i, :]

        return r

    # CPU version of double exponential filter
    @staticmethod
    def double_exponential_filter_mimo(spike_train, dt, tau_r, tau_d):
        """

        :param spike_train: spike_train of one neuron; shape: (n)
        :param dt: time step
        :param tau_r: synaptic rise time
        :param tau_d: synaptic decay time
        :return: filtered spike_train
        """
        r = Variable(torch.zeros(spike_train.shape[0], spike_train.shape[1], spike_train.shape[2]))
        h = torch.zeros(spike_train.shape[0], spike_train.shape[1], spike_train.shape[2])
        for inp in range(spike_train.shape[0]):
            for n in range(1, spike_train.shape[1]):
                h[inp, n, :] = h[inp, n - 1, :] * (1 - dt / tau_r) + (dt / (tau_r * tau_d)) * spike_train[inp, n - 1, :]
                r[inp, n, :] = r[inp, n - 1, :] * (1 - dt / tau_d) + dt * h[inp, n, :]

        return r

    # GPU version of double exponential filter
    @staticmethod
    def double_exponential_filter_mimo_cuda(spike_train, dt, tau_r, tau_d):
        """

        :param spike_train: spike_train
        :param dt: time step
        :param tau_r: synaptic rise time
        :param tau_d: synaptic decay time
        :return: filtered spike_train
        """

        r = Variable(torch.zeros(spike_train.shape[0], spike_train.shape[1], spike_train.shape[2]).cuda())
        h = torch.zeros(spike_train.shape[0], spike_train.shape[1], spike_train.shape[2]).cuda()
        for inp in range(spike_train.shape[0]):  # number of classes
            for n in range(1, spike_train.shape[1]):  # number of time steps
                h[inp, n, :] = h[inp, n - 1, :] * (1 - dt / tau_r) + (dt / (tau_r * tau_d)) * spike_train[inp, n - 1, :]
                r[inp, n, :] = r[inp, n - 1, :] * (1 - dt / tau_d) + dt * h[inp, n, :]
        return r
