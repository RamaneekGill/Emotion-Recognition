from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pylab
import time

class LogisticRegression():
	def __init__(self, train_dataset, validation_dataset):
		# The default hyperparameters of the model
		self.setLearningRate(0.001)
		self.setEpochs(100)
		self.setBatchSize(50)
		self.setDisplayStep(1)

		# The datasets to use
		self.train_dataset = train_dataset
		self.validation_dataset = validation_dataset

		# The data to feed to the model
		self.num_pixels = self.train_dataset.inputs().shape[1]
		self.num_classes = self.train_dataset.targets().shape[1]
		self.x = tf.placeholder(tf.float32, [None, self.num_pixels])
		self.y = tf.placeholder(tf.float32, [None, self.num_classes])

		# The structure of the default model
		self.W = tf.Variable(tf.random_normal([self.num_pixels, self.num_classes], stddev=1))
		self.b = tf.Variable(tf.random_normal([self.num_classes], stddev=1))
		self.model = tf.matmul(self.x, self.W) + self.b
		self.useSoftmaxActivation() # tf.nn.softmax(self.model)
		self.useSoftmaxLoss() # self.y * tf.log(self.activation)
		self.useReducedMeanCost() # -tf.reduce_sum(self.loss)
		self.useGradientDescent()

		# Initialize the tensorflow session
		self.start = time.time()
		self.init = tf.initialize_all_variables()
		self.sess = tf.Session()
		self.sess.run(self.init)
		self.saver = tf.train.Saver()


	def setLearningRate(self, rate):
		self.learningRate = rate

	def getLearningRate(self):
		return self.learningRate

	def setEpochs(self, epochs):
		self.epochs = epochs

	def getEpochs(self):
		return self.epochs

	def setBatchSize(self, size):
		self.batchSize = size

	def getBatchSize(self):
		return self.batchSize

	def setDisplayStep(self, step):
		self.displayStep = step

	def getDisplayStep(self):
		return self.displayStep

	def getWeights(self):
		return self.W

	def getBias(self):
		return self.b

	def getActivation(self):
		return self.activation

	def useSigmoidActivation(self):
		self.using_activation = 'sigmoid'
		self.activation = tf.nn.sigmoid(self.model)

	def useSoftmaxActivation(self):
		self.using_activation = 'softmax'
		self.activation = tf.nn.softmax(self.model)

	def getModel(self):
		return self.model

	def getLoss(self):
		return self.loss

	def useSoftmaxLoss(self):
		self.using_loss = 'cross_entropy'
		self.loss = tf.nn.softmax_cross_entropy_with_logits(self.model, self.y)

	def useSigmoidLoss(self):
		self.using_loss = 'sigmoid'
		self.loss = tf.nn.sigmoid_cross_entropy_with_logits(self.model, self.y)

	def useReducedSumCost(self):
		self.using_cost = 'reduced_sum'
		self.cost = tf.reduce_sum(self.loss)

	def useReducedMeanCost(self):
		self.using_cost = 'reduced_mean'
		self.cost = tf.reduce_mean(self.loss)

	def useGradientDescent(self):
		self.using_optimizer = 'gradient_descent'
		self.optimizer = tf.train.GradientDescentOptimizer(self.getLearningRate()).minimize(self.cost)

	def getAccuracy(self, dataset):
		correct_prediction = tf.equal(tf.argmax(self.activation, 1), tf.argmax(self.y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return self.sess.run(accuracy, feed_dict={self.x: dataset.inputs(), self.y: dataset.targets()})

	def train(self):
		self.avg_cost = 0.0
		num_batches = int(self.train_dataset.num_examples() / self.getBatchSize())
		self.train_set_accuracies = []
		self.validation_set_accuracies = []
		self.cross_entropies = []

		for epoch in range(self.getEpochs()):
			print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(self.avg_cost)
			train_set_accuracy = self.getAccuracy(self.train_dataset)
			validation_set_accuracy = self.getAccuracy(self.validation_dataset)
			print("Accuracy on train set is {} \t validation set: {}".format(train_set_accuracy, validation_set_accuracy))

			self.train_set_accuracies.append(train_set_accuracy)
			self.validation_set_accuracies.append(validation_set_accuracy)
			self.cross_entropies.append(self.avg_cost)

			for batch in range(num_batches):
				batch_xs, batch_ys = self.train_dataset.next_batch(self.getBatchSize())
				self.sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys})
				self.avg_cost += self.sess.run(self.cost, feed_dict={self.x: batch_xs, self.y: batch_ys}) / num_batches

if __name__ == "__main__":
	print('Starting the script!')
	import input_data
	datasets = input_data.read_data_sets()

	logreg = LogisticRegression(datasets.train_set, datasets.validation_set)
	logreg.train()

	plt.figure(1)
	plt.clf()
	plt.plot(range(logreg.getEpochs()), logreg.train_set_accuracies, '-r', label='training set accuracy')
	plt.plot(range(logreg.getEpochs()), logreg.validation_set_accuracies, '-b', label='validation set accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Classification Error')
	plt.title('Classification Error While Training Using Softmax')
	plt.legend()
	plt.show()
	raw_input('Press Enter to exit.')
