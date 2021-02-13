import sys

sys.path.append('../')
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from spike_filters import SpikeFilter
from framework_utils import *


class RNNSpikeSignal(object):
    file_name = 'mi_'
    create_dir("saveddata")
    create_dir("syntheticdata")

    def __init__(self, neurons, n_steps, n_epochs, desired_spike_signal, noise_neurons, noise_factor, freq=20, lr=2e-3,
                 perturb=False, syn_trials=10, eta=1, last_epoch=None):

        self.neurons = neurons  # list of numbers of neuron in all layers (input, hidden, output)
        self.n_steps = n_steps  # number of simulation steps
        self.n_epochs = n_epochs  # number of epochs in raining
        self.desired_spike_signal = desired_spike_signal  # target signal
        self.n_layers = len(neurons)  # number of layers
        self.time_step = 1e-3  # time step duration
        self.freq = freq
        self.V = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device cpu or gpu
        self.classes = len(desired_spike_signal)  # find the number of classes
        self.dtype = torch.float

        self.batch_Size = self.classes
        # neuron parameter
        self.tau_mem = 10e-3
        self.tau_syn = 5e-3
        self.alpha = float(np.exp(-self.time_step / self.tau_syn))
        self.beta = float(np.exp(-self.time_step / self.tau_mem))

        self.WF = []  # forward weights
        self.WR = []  # filter weigths
        self.WT = []  # perturbation weights
        self.WN = []
        self.n_hidden_layers = len(neurons) - 2  # number of hidden layers
        self.loss_fn = nn.MSELoss()  # MSE loss function
        self.lr = lr  # learning rate
        self.input_current = self.get_poisson_input()  # generate input for each class

        self.n_outputs = desired_spike_signal[0].shape[0]  # number of output channels
        self.eta = eta
        self.noise_factor = noise_factor  # noise firing rate to vary
        self.noise_neurons = noise_neurons  # noise neurons to vary
        self.syn_trials = syn_trials
        self.tau_d = 2e-3
        self.tau_r = 3e-3
        self.mem_rec_layers = []  # membrane voltage of each layer
        self.spk_rec_layers = []  # spike record of each layer

        self.perturb = perturb
        self.last_epoch = last_epoch

    def get_poisson_input(self):
        """

        :return: input layer spike_train of dimension: batchsize x n_steps x #of input neurons
        """
        try:
            with open(RNNSpikeSignal.file_name + '_input_spike_data', 'rb') as f:
                x_data = pickle.load(f)
            f.close()
            print("loading input_data")
        except:
            prob = self.freq * self.time_step
            mask = torch.rand((self.batch_Size, self.n_steps, self.neurons[0]), dtype=self.dtype, device=self.device)
            x_data = torch.zeros((self.batch_Size, self.n_steps, self.neurons[0]), dtype=self.dtype, device=self.device,
                                 requires_grad=False)
            x_data[mask < prob] = 1.0
            with open(RNNSpikeSignal.file_name + '_input_spike_data', 'wb') as f:
                pickle.dump(x_data, f)
            f.close()
        return x_data

    # for data generation
    def get_noise_input(self):
        """

        :return: noise layer spike_train of dimension: batchsize x n_steps x #of noise neurons
        """
        prob = self.noise_factor * self.freq * self.time_step
        # print(prob)
        mask = torch.rand((self.batch_Size, self.n_steps, self.noise_neurons), dtype=self.dtype, device=self.device)
        x_data = torch.zeros((self.batch_Size, self.n_steps, self.noise_neurons), dtype=self.dtype, device=self.device,
                             requires_grad=False)
        # x_data[mask < prob] = 1.0
        x_data[mask < prob] = 1.0
        self.noise_spikes_count = torch.sum(x_data)
        return x_data

    def plot_spike_raster(self, data):
        """

        :param data: 2D array : n_steps x neurons
        :return: raster plot
        """
        plt.imshow(data.t(), cmap=plt.cm.gray_r, aspect="auto")
        plt.xlabel("Time (ms)")
        plt.ylabel("Unit")

    def init_network(self):
        weight_scale = 7 * (1.0 - self.beta)  # this should give us some spikes to begin with
        for layer in range(self.n_layers - 1):
            self.WF.append(
                torch.empty((self.neurons[layer], self.neurons[layer + 1]), device=self.device, dtype=self.dtype,
                            requires_grad=True))
            torch.nn.init.normal_(self.WF[layer], mean=0.0, std=weight_scale / np.sqrt(self.neurons[layer]))
        self.WR = torch.empty((self.neurons[-1], self.n_outputs), device=self.device, dtype=self.dtype,
                              requires_grad=True)
        torch.nn.init.normal_(self.WR, mean=0.0, std=weight_scale / np.sqrt(self.neurons[-1]))

        for layer in range(1):
            self.WN.append(torch.empty((self.noise_neurons, self.neurons[-1]), device=self.device, dtype=self.dtype,
                                       requires_grad=False))
            torch.nn.init.normal_(self.WN[layer], mean=0.0, std=weight_scale / np.sqrt(self.neurons[-1]))
        print("init done")

    def spike_fn(self, x):
        out = torch.zeros_like(x)
        out[x > 0] = 1.0
        return out

    def forward(self):
        self.mem_rec_layers = []
        self.spk_rec_layers = []
        input_data = self.input_current
        # computer layer wise activity
        for layer in range(self.n_layers - 1):
            syn = torch.zeros((self.batch_Size, self.neurons[layer + 1]), device=self.device, dtype=self.dtype)
            mem = torch.zeros((self.batch_Size, self.neurons[layer + 1]), device=self.device, dtype=self.dtype)
            mem_rec = [mem]
            spk_rec = [mem]

            h = torch.einsum("abc,cd->abd", (input_data, self.WF[layer]))
            if layer == self.n_layers - 2 and self.perturb:
                noise_input = self.get_noise_input()
                nh = torch.einsum("abc,cd->abd", (noise_input, self.WN[0]))
            else:
                noise_input = torch.zeros((self.batch_Size, self.n_steps, self.noise_neurons), dtype=self.dtype,
                                          device=self.device,
                                          requires_grad=False)
                nh = torch.einsum("abc,cd->abd", (noise_input, self.WN[0]))

            # itetrate througho time steps
            for t in range(self.n_steps):
                mthr = mem - 1.0
                out = self.spike_fn(mthr)  # estimat membrane threshold
                rst = torch.zeros_like(mem)
                c = (mthr > 0)
                # rst[c] = torch.ones_like(mem)[c]
                rst[c] = mem[c]
                rst1 = torch.ones_like(mem)
                rst1[c] = torch.zeros_like(mem)[c]

                ## current estimation
                if layer == self.n_layers - 2:  # add noise if needed to the last layer
                    new_syn = self.alpha * syn + h[:, t] + self.eta * nh[:, t]
                else:
                    new_syn = self.alpha * syn + h[:, t]

                ## membrane voltage update
                new_mem = self.beta * mem + rst1 * syn - rst
                mem = new_mem
                syn = new_syn
                mem_rec.append(mem)
                spk_rec.append(out)

            mem_rec = torch.stack(mem_rec, dim=1)
            spk_rec = torch.stack(spk_rec, dim=1)
            input_data = spk_rec
            self.mem_rec_layers.append(mem_rec)
            self.spk_rec_layers.append(spk_rec)
        return spk_rec

    def train(self):
        self.spike_fn = SuperSpike.apply
        self.init_network()
        params = [w for w in self.WF] + [self.WR]  # list weight paramerter
        optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999))
        loss = []
        for epoch in range(self.n_epochs):
            output_spike_train = self.forward()  # obtain output spike train from spike output layer
            output_spike_train = output_spike_train[:, 1:, :]

            # apply double exponential filter to the output spike train
            if torch.cuda.is_available():
                output_filter = SpikeFilter.double_exponential_filter_mimo_cuda(output_spike_train, dt=self.time_step,
                                                                                tau_r=self.tau_r,
                                                                                tau_d=self.tau_d).cuda()
            else:
                output_filter = SpikeFilter.double_exponential_filter_mimo(output_spike_train, dt=self.time_step,
                                                                           tau_r=self.tau_r,
                                                                           tau_d=self.tau_d)

            # transform filtered signal to output eeg signal
            output_signal = torch.einsum("abc,cd->abd", (output_filter, self.WR))
            output_signal = output_signal.transpose(1, 2)  # batch(class) x channel x time

            # self.desired_spike_signal -- class(batch) x channel x time
            loss_val = 1000 * 20 * self.loss_fn(torch.tensor(self.desired_spike_signal).type(torch.DoubleTensor),
                                                output_signal.type(torch.DoubleTensor))

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print("epoch: {}  loss: {}".format(epoch, loss_val.detach().numpy()))
                with open("saveddata/" + RNNSpikeSignal.file_name + '_outpus_signal_epoch_{}.p'.format(epoch),
                          'wb') as f:
                    pickle.dump(output_signal, f)
                f.close()
                str1 = ''.join(str(e) for e in self.neurons)
                with open("saveddata/" + RNNSpikeSignal.file_name + "losses.p", "wb") as f:
                    pickle.dump(loss, f)
                f.close()
                torch.save(self.WF, "saveddata/" + RNNSpikeSignal.file_name + "_epoch_{}_WF_{}".format(epoch, str1))
                torch.save(self.WR, "saveddata/" + RNNSpikeSignal.file_name + "_epoch_{}_WR_{}".format(epoch, str1))

    ## create artificial trials
    def create_trials(self):
        output_signal = []
        self.spike_fn = SuperSpike.apply
        self.init_network()
        self.done_train = True
        output_spike = []

        # load saved weights
        try:
            str1 = ''.join(str(e) for e in self.neurons)
            self.WF = torch.load(
                "saveddata/" + RNNSpikeSignal.file_name + "_epoch_{}_WF_".format(self.last_epoch) + str1)
            self.WR = torch.load("savedata/" + RNNSpikeSignal.file_name + "_epoch_{}_WR_".format(self.last_epoch) + str1)
            print("loading saved weights")
        except:
            print("Load last saved epoch weights")

        ## genereate synthetic samples
        for i in range(self.syn_trials):
            output_spike_train = self.forward()
            output_spike_train = output_spike_train[:, 1:, :]
            if torch.cuda.is_available():
                output_filter = SpikeFilter.double_exponential_filter_mimo_cuda(output_spike_train, dt=self.time_step,
                                                                                tau_r=self.tau_r,
                                                                                tau_d=self.tau_d).cuda()
            else:
                output_filter = SpikeFilter.double_exponential_filter_mimo(output_spike_train, dt=self.time_step,
                                                                           tau_r=self.tau_r,
                                                                           tau_d=self.tau_d)
            out = torch.einsum("abc,cd->abd", (output_filter, self.WR))
            out = out.transpose(1, 2)
            output_signal.append(out)
            output_spike.append(output_spike_train)
        with open("syntheticdata/" + RNNSpikeSignal.file_name + '_noise_{}'.format(
                self.noise_neurons) + '_noise_factor_{}'.format(self.noise_factor) + '_synth_trials.p', 'wb') as f:
            pickle.dump(output_signal, f)
        f.close()


class SuperSpike(torch.autograd.Function):
    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SuperSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad
