import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

# load dataset
cls1_data = sio.loadmat('B0403TClass1.mat')['data'][[0, 2], :, :]  # get C3 C4 channels
cls1_data = cls1_data.transpose(2, 0, 1)
labels1 = np.zeros(cls1_data.shape[0])

cls2_data = sio.loadmat('B0403TClass2.mat')['data'][[0, 2], :, :]  # get C3 C4 channels
cls2_data = cls2_data.transpose(2, 0, 1)
labels2 = np.ones(cls2_data.shape[0])

# pre-process data : remove mean
def remove_mean(data):
    x = np.zeros_like(data)
    for tr in range(data.shape[0]):
        for ch in range(data.shape[1]):
            sample = data[tr, ch, :]
            x[tr, ch, :] = sample - np.mean(sample)
    return x


cls1_data = remove_mean(cls1_data)
cls2_data = remove_mean(cls2_data)

#data concatenation
data = np.concatenate((cls1_data, cls2_data), 0).astype(np.float64)
labels = np.concatenate((labels1, labels2)).astype(np.int64)

#create csp object
csp = CSP(n_components=10, reg=0.01, log=False, norm_trace=True)
scores = []

#tranform all data to csp features
csp.fit(data, labels)
train_data = csp.transform(data)
train_labels = labels

#instantiate LDA classifier
clf = LinearDiscriminantAnalysis()
clf.fit(train_data, train_labels)

# Estimate the probability of each sample to be labelled
probs = clf.predict_proba(train_data)
trials_n1 = len(labels1)
trials_n2 = len(labels2)

# seprate class probabilites
cls1_probs = probs[:trials_n1, 0]
cls2_probs = probs[trials_n1:, 1]

# obtain index of maximum probability of each class
cls1_inds = np.argmax(cls1_probs)
cls2_inds = np.argmax(cls2_probs)


# extract sample with
best_cls1 = np.array([cls1_data[cls1_inds, :, :]])
best_cls2 = np.array([cls2_data[cls2_inds, :, :]])

#cross check the probability of the best sample
test_data = np.concatenate((best_cls1, best_cls2), 0).astype(np.float64)
test_labels = np.array([0, 1]).astype(np.int64)

test_feat = csp.transform(test_data)
outs = clf.predict_proba(test_feat)
print(outs[0][0], outs[1][1])

# save best samples
np.save('B0403TClass1_best', best_cls1[0, :, :])
np.save('B0403TClass2_best', best_cls1[0, :, :])
