import pandas as pandas
import numpy as np
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix

NUM_CLASSES = 4
EPOCHS = 10000

red_data = pandas.read_csv("datasets/winequality-red.csv", sep=';').to_numpy(dtype=np.float128)
white_data = pandas.read_csv("datasets/winequality-white.csv", sep=';').to_numpy(dtype=np.float128)

#print(red_data.shape)
#print(white_data.shape)

np.random.shuffle(red_data)
np.random.shuffle(white_data)

rng     = np.random.default_rng(12345)
theta   = rng.random((12,NUM_CLASSES))#np.random.rand(12,NUM_CLASSES) #weigth array. each class in a row and each featrue is a collum

def bucketize(y): #reduces data to 3 classes
    y = np.asarray(y, dtype=int)
    # np.digitize splits at 5, 7 and produces 0, 1, 2
    return np.digitize(y, bins=[4,6,8], right=False)

# Red wine
red_targets_raw     = red_data[:, -1]     
print(np.unique(red_targets_raw, return_counts=True))

red_targets         = bucketize(red_targets_raw) 
print(np.unique(red_targets, return_counts=True))
hot_red_targets     = np.eye(NUM_CLASSES)[red_targets]

red_samples         = np.ones((red_data.shape[0], red_data.shape[1]))
red_samples[:, 1:]  = red_data[:, :-1]

#Splits into testing and trainging data
red_testing         = red_samples[:500, :]
red_training        = red_samples[500:,:]

hot_red_testing     = hot_red_targets[:500,:]
hot_red_training    = hot_red_targets[500:,:]

# White wine
white_targets_raw   = white_data[:, -1]
print(np.unique(white_targets_raw, return_counts=True))

white_targets       = bucketize(white_targets_raw)
print(np.unique(white_targets, return_counts=True))

hot_white_targets   = np.eye(NUM_CLASSES)[white_targets]

white_samples       = np.ones((white_data.shape[0], white_data.shape[1]))
white_samples[:, 1:] = white_data[:, :-1]

# splitting into testing and trainging data
white_testing       = white_samples[:1000, :]
white_training      = white_samples[1000:, :]

hot_white_testing   = hot_white_targets[:1000, :]
hot_white_training  = hot_white_targets[1000:, :]




def soft_max(weights, samples):
    scores          = samples @ weights
    scores          -= scores.max(axis=1, keepdims=True) #
    e               = np.exp(scores)

    return e / e.sum(axis=1,keepdims=True)

def regression_loop(weights, samples, targets):
    mu = 0.0001
    n = samples.shape[0]
    for i in range(EPOCHS):
        y           = soft_max(weights, samples)
        gradient    = (samples.T @ (y - targets)) / n
        weights     -= mu*gradient

        if i % 200 == 0:
            #loss = -np.mean(np.sum(targets * np.log(y + 1e-15), axis=1))
            loss = np.mean(np.sum((targets - y)*(targets - y), axis=1))
            print(f"epoch {i:4d} | loss {loss:.4f}")

    return weights



theta       = regression_loop(theta, white_training, hot_white_training)

predicted   = np.argmax(white_testing @ theta, axis = 1)

white_testing_targets = white_targets[:1000]

cm          = confusion_matrix(white_testing_targets, predicted, labels=[0,1,2,3])

print(cm)


theta       = rng.random((12,NUM_CLASSES))

theta       = regression_loop(theta, red_training, hot_red_training) 

predicted   = np.argmax(red_testing @ theta, axis = 1)

red_testing_targets = red_targets[:500]

cm          = confusion_matrix(red_testing_targets, predicted, labels=[0,1,2,3])

print(cm)