import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt



# Preprocess the data before building model
def load_data():

	# Load data
	labels = np.load('labels.npy')
	inputs = np.load('images.npy')


	# Convert input matrix to vectors
	#flattened_input = np.reshape(inputs, (6500,784))

	# Convert output labels to categorical vectors
	encoded_labels = to_categorical(labels, num_classes=10)

	# Bad input output
	# TODO: Stratified sampling
	itrain, ivalidate, itest = inputs[:int(0.6*6500), :], inputs[int(0.6*6500):int(0.75*6500), :], inputs[int(0.75*6500):, :]
	ltrain, lvalidate, ltest = encoded_labels[:int(0.6*6500), :], encoded_labels[int(0.6*6500):int(0.75*6500), :], encoded_labels[int(0.75*6500):6500, :]

	# Return input (train, validate, test), output (train, validate, test)
	return itrain, ivalidate, itest, ltrain, lvalidate, ltest


