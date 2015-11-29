from __future__ import division
import tensorflow as tf
import numpy as np
import time

class LogisticRegression():
	def __init__(self):
		self.learningRate = 0.01
		self.epochs = 1000
		self.batchSize = 100

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

	def getWeights(self):
		return self.W

	def getBias(self):
		return self.b

	def getActivation(self):
		return self.activation
