import numpy as np
from keras.utils import to_categorical



# Preprocess the data before building model
def load_data():

	# Load data
	labels = np.Load('labels.npy')
	inputs = np.Load('images.npy')

	# Convert input matrix to vectors
	flattened_input = np.reshape(inputs, (6500,784))

	# Convert output labels to categorical vectors
	encoded_labels = to_categorical(labels, num_classes=10)

	# Bad input output
	# TODO: Stratified sampling
	itrain, ivalidate, itest = flattened_input[(0:(0.6*6500)), :], flattened_input[((0.6*6500):(0.75*6500)), :], flattened_input[((0.75*6500):6500), :]
	ltrain, lvalidate, ltest = encoded_labels[(0:(0.6*6500)), :], encoded_labels[((0.6*6500):(0.75*6500)), :], encoded_labels[((0.75*6500):6500), :]

	# Return input (train, validate, test), output (train, validate, test)
	return itrain, ivalidate, itest, ltrain, lvalidate, ltest


