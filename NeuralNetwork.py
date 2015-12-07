from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import heapq
import pylab
import time

### Used this source as reference material:
# https://github.com/nlintz/TensorFlow-Tutorials/blob/master/5_convolutional_net.py

def testNetwork(num_hidden_units, decay, retain_input, retain_hidden):
	import input_data
	datasets = input_data.read_data_sets()

	NUM_PIXELS = datasets.train_set.inputs().shape[1]
	NUM_CLASSES = datasets.train_set.targets().shape[1]

	### Setup input pipelines
	x = tf.placeholder(tf.float32, [None, NUM_PIXELS])
	y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

	# The hyperparameters of the model
	NUM_HIDDEN_UNITS = num_hidden_units
	LEARNING_RATE = 0.005
	DECAY = decay
	MOMENTUM = 0.1
	EPOCHS = 3000
	BATCH_SIZE = 100
	RETAIN_INPUT = retain_input
	RETAIN_HIDDEN = retain_hidden

	### Construct model
	hidden_layer_1_weights = tf.Variable(tf.random_normal([NUM_PIXELS, NUM_HIDDEN_UNITS], stddev=0.01))
	hidden_layer_2_weights = tf.Variable(tf.random_normal([NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS], stddev=0.01))
	output_layer_weights = tf.Variable(tf.random_normal([NUM_HIDDEN_UNITS, NUM_CLASSES], stddev=0.01))

	p_retain_input = tf.placeholder(tf.float32)
	p_retain_hidden = tf.placeholder(tf.float32)

	x = tf.nn.dropout(x, p_retain_input) # use dropout training

	hidden_layer_1 = tf.nn.relu(tf.matmul(x, hidden_layer_1_weights))
	hidden_layer_1 = tf.nn.dropout(hidden_layer_1, p_retain_hidden) # use dropout training

	hidden_layer_2 = tf.nn.softmax(tf.matmul(hidden_layer_1, hidden_layer_2_weights))
	hidden_layer_2 = tf.nn.dropout(hidden_layer_2, p_retain_hidden) # use dropout training

	model = tf.matmul(hidden_layer_2, output_layer_weights)

	# Define the learning process
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))
	optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, DECAY).minimize(cost)
	correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# train_x = datasets.train_set.inputs()
	# train_y = datasets.train_set.targets()
	# print(train_x.shape, train_y.shape)
	# print(train_x)
	# print('sdfsdfs')
	# print(train_y)
	# exit()

	# Create a session and initialize all variables
	print('Starting session')
	sess = tf.Session()
	init = tf.initialize_all_variables()
	sess.run(init)
	avg_cost = 0.0

	train_accuracies = []
	validation_accuracies = []

	# Train the model
	for epoch in range(EPOCHS):
		num_batches = int(datasets.train_set.num_examples() / BATCH_SIZE)
		for batch in range(num_batches):
			batch_xs, batch_ys = datasets.train_set.next_batch(BATCH_SIZE)
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, p_retain_input: RETAIN_INPUT, p_retain_hidden: RETAIN_HIDDEN})
			avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, p_retain_input: RETAIN_INPUT, p_retain_hidden: RETAIN_HIDDEN}) / num_batches
		# print avg_cost

		# Print the accuracies on training and validation sets
		train_accuracy = sess.run(accuracy, feed_dict={x: datasets.train_set.inputs(), y: datasets.train_set.targets(), p_retain_input: 1, p_retain_hidden: 1})
		train_accuracies.append(train_accuracy)
		validation_accuracy = sess.run(accuracy, feed_dict={x: datasets.validation_set.inputs(), y: datasets.validation_set.targets(), p_retain_input: 1, p_retain_hidden: 1})
		validation_accuracies.append(validation_accuracy)
		# if epoch % 100 == 0:
		print("Epoch: {} \t Train accuracy: {} \t Validation accuracy: {}".format(epoch, train_accuracy, validation_accuracy))

	return heapq.nlargest(3, np.array(validation_accuracies))


if __name__ == "__main__":
	top_3_validation_accuracies = []
	num_hidden_units_intervals = [5, 10, 25, 50, 100, 200, 400]
	decay_intervals = [0.9, 0.6, 0.3, 0.1]
	retain_input_intervals = [0.9, 0.6, 0.3, 0.1]
	retain_hidden_intervals = [0.9, 0.6, 0.3, 0.1]

	accuracies = testNetwork(100, 0.4, 0.3, 0.3)
	print(accuracies)
	exit()

	iteration = 0
	for num_hidden_units in num_hidden_units_intervals:
		for decay in decay_intervals:
			for retain_input in retain_input_intervals:
				for retain_hidden in retain_hidden_intervals:
					iteration += 1
					accuracies = testNetwork(num_hidden_units, decay, retain_input, retain_hidden)
					top_3_validation_accuracies.append(accuracies)
					print("hidden units: {} \t decay: {} \t retain input: {} \t retain hidden: {}"
						.format(num_hidden_units, decay, retain_input, retain_hidden))
					print(iteration, accuracies)


	print("THESE ARE THE FINAL ACCURACIES")
	print(accuracies)
