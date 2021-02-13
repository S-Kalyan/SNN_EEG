import sys

sys.path.append('../')
from mi_bci import RNNSpikeSignal
import numpy as np

n_epochs = 10001  # number of epoch to run
#
layer_wise_neurons = [600, 1500, 400]  # number of nodes in each layer layers of spiking network
dt = 1e-3

# load templates of each class
class1_eeg = np.load('data/B0403TClass1_best.npy')
class2_eeg = np.load('data/B0403TClass2_best.npy')

desired_signal = [class1_eeg, class2_eeg]

n_steps = len(class1_eeg[0])

model = RNNSpikeSignal(neurons=layer_wise_neurons, n_steps=n_steps, n_epochs=n_epochs,
                       desired_spike_signal=desired_signal, noise_neurons=10, noise_factor=0.5, freq=20,
                       lr=1e-4, last_epoch=None, perturb=False)
model.train()
