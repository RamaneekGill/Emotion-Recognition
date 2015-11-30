from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pylab
import time

if __name__ == "__main__":
	import input_data
	datasets = input_data.read_data_sets()

	NUM_PIXELS = datasets.train_set.inputs().shape[1]
	NUM_CLASSES = datasets.train_set.targets().shape[1]

	### Setup input pipelines
	x = tf.placeholder(tf.float32, tf.Variable(tf.random_normal([None, NUM_PIXELS], stddev=0.01)))
	y = tf.placeholder(tf.float32, tf.Variable(tf.random_normal([None, NUM_CLASSES], stddev=0.01)))

	# The hyperparameters of the model
	NUM_HIDDEN_UNITS = 500
	LEARNING_RATE = 0.001
	DECAY = 0.9
	MOMENTUM = 0
	EPOCHS = 100
	BATCH_SIZE = 100

	### Construct model
	hidden_layer_1_weights = tf.Variable(tf.random_normal([NUM_PIXELS, NUM_HIDDEN_UNITS], stddev=0.01))
	hidden_layer_2_weights = tf.Variable(tf.random_normal([NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS], stddev=0.01))
	output_layer_weights = tf.Variable(tf.random_normal([NUM_HIDDEN_UNITS, NUM_CLASSES], stddev=0.01))

	p_retain_input = tf.placeholder(tf.float32)
	p_retain_hidden = tf.placeholder(tf.float32)

	x = tf.nn.dropout(x, p_retain_input) # use dropout training

	hidden_layer_1 = tf.nn.relu(tf.matmul(x, hidden_layer_1_weights))
	hidden_layer_1 = tf.nn.dropout(hidden_layer_1, p_retain_hidden) # use dropout training

	hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, hidden_layer_2_weights))
	hidden_layer_2 = tf.nn.dropout(hidden_layer_2, p_retain_hidden) # use dropout training

	model = tf.matmul(hidden_layer_2, output_layer_weights)

	# Define the learning process
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))
	optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, DECAY, MOMENTUM)
	correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Create a session and initialize all variables
	sess = tf.Session()
	init = tf.initialize_all_variables()
	sess.run(init)

	# Train the model
	for epoch in range(EPOCHS):
		num_batches = int(datasets.train_set.num_examples() / BATCH_SIZE)
		for batch in range(num_batches):
			batch_xs, batch_ys = datasets.train_set.next_batch(BATCH_SIZE)
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / num_batches

		# Print the accuracies on training and validation sets
		train_accuracy = sess.run(accuracy, feed_dict={x: datasets.train_set.inputs(), y: datasets.train_set.targets()})
		validation_accuracy = sess.run(accuracy, feed_dict={x: datasets.validation_set.inputs(), y: datasets.validation_set.targets()})
		print("Train accuracy: {} \t Validation accuracy: {}".format(train_accuracy, validation_accuracy))
