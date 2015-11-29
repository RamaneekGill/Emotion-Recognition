import tensorflow as tf
import scipy.io as sio
import numpy as np

def flatten_images(images):
	num_images = len(images[0][0])
	flattened_images = []

	for i in range(num_images):
		flattened_images.append(images[:,:,i].flatten())

	return np.array(flattened_images)

def splitSet(train_images, train_identities, test_set_length):
	# Find how many images each identity has
	identities = dict()
	for identity in train_identities:
		identities[identity[0]] = identities.get(identity[0], 0) + 1

	# Find which identities to use in validation set
	identities_to_use_as_validation = []
	count = 0
	for identity in train_identities:
		if count + identities[identity[0]]:
			continue
		count += identities[identity[0]]
		identities_to_use_as_validation.append(identity[0])


	train_set = []
	validation_set = []

	for i in range(len(train_images)):
		if train_identities[i][0] in identities_to_use_as_validation:
			validation_set.append(train_images[i])
		else:
			train_set.append(train_images[i])

	return np.array(train_set), np.array(validation_set)


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
test_images = flatten_images(test_images)
train_images = flatten_images(train_images)

# Split train into validation set of size ~ test_set_length
train_images, validation_images = splitSet(train_images, train_identities, test_set_length)

print(test_images.shape, train_images.shape, validation_images.shape)
