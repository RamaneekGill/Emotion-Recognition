import tensorflow as tf
import scipy.io as sio
import numpy as np

def flattenImages(images):
	num_images = len(images[0][0])
	flattened_images = []

	for i in range(num_images):
		flattened_images.append(images[:,:,i].flatten())

	return np.array(flattened_images)

def splitSet(train_images, train_labels, train_identities, test_set_length):
	# Find how many images each identity has
	identities = dict()
	for identity in train_identities:
		identities[identity[0]] = identities.get(identity[0], 0) + 1

	# Find which identities to use in validation set
	identities_to_use_as_validation = []
	count = 0
	for identity in train_identities:
		if count + identities[identity[0]] > test_set_length:
			continue
		count += identities[identity[0]]
		identities_to_use_as_validation.append(identity[0])


	train_inputs = []
	train_targets = []
	validation_inputs = []
	validation_targets = []

	for i in range(len(train_images)):
		if train_identities[i][0] in identities_to_use_as_validation:
			validation_inputs.append(train_images[i])
			validation_targets.append(train_labels[i])
		else:
			train_inputs.append(train_images[i])
			train_targets.append(train_labels[i])

	return np.array(train_inputs), np.array(train_targets), np.array(validation_inputs), np.array(validation_targets)

def convertToOneHot(labels, num_classes):
	one_hot = np.zeros((len(labels), num_classes))

	for i in range(len(labels)):
		one_hot[i][labels[i][0]-1] = 1

	return one_hot

NUM_CLASSES = 7

# Load the training data
mat_contents = sio.loadmat('labeled_images.mat')
train_labels = mat_contents['tr_labels']
train_identities = mat_contents['tr_identity']
train_images = mat_contents['tr_images']

# Load the test data
mat_contents = sio.loadmat('public_test_images.mat')
test_images = mat_contents['public_test_images']
test_set_length = len(test_images[0][0])

# Flatten images
test_images = flattenImages(test_images)
train_images = flattenImages(train_images)

# Split train into validation set of size ~ test_set_length
train_images, train_labels, validation_images, validation_labels = splitSet(train_images,
	train_labels, train_identities, test_set_length)

# Convert labels to one hot vectors
train_labels = convertToOneHot(train_labels, NUM_CLASSES)
validation_labels = convertToOneHot(validation_labels, NUM_CLASSES)
