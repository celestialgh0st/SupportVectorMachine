import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class BinarySVM():
    def __init__(self, X, y, gd_alpha=0.001, alpha=10., batch_size=200, epoch=100):
        """This fuction defines the graph of the SVM model"""
        # define training values
        self.X = X
        self.y = y
        self.data = tf.placeholder(shape=[None, len(self.X[0])], dtype=tf.float32, name='data')
        self.target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='target')
        self._optimize = None
        self._prediction = None
        self._accuracy = None
        self.gd_alpha = gd_alpha
        self.alpha = alpha
        self.batch_size = batch_size
        self.epoch = epoch
        num_of_attr = len(self.X[0])
        self.__weights = tf.Variable(tf.random_normal(shape=[num_of_attr, 1]), name='Weights')
        self.__bias = tf.Variable(tf.random_normal(shape=[1, 1]), name='bais')

    @property
    def prediction(self):
        """If the predict functions runs for the first time it will set all the weights and biases and return _predict variable """
        if self._prediction is None:
            model_output = tf.subtract(tf.matmul(self.data, self.__weights), self.__bias, name='Model_Output')
            self._prediction = model_output
        return self._prediction

    @property
    def optimize(self):
        """This function is used to optimize the weights and bias of the model """
        if self._optimize is None:
            l2_norm = tf.reduce_sum(tf.square(self.__weights))
            classificaton_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(self.target, self.prediction))))
            loss = tf.add(classificaton_term, tf.multiply(self.alpha, l2_norm))
            myopt = tf.train.GradientDescentOptimizer(0.01)
            train_step = myopt.minimize(loss)
            self._optimize = train_step
        return self._optimize

    @property
    def accuracy(self):
        """Calculates the accuracy based on given attributes and target value"""
        if self._accuracy is None:
            prediction = tf.sign(self._prediction)
            self._accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, self.target), tf.float32))
        return self._accuracy

    def fit(self):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        acc_temp = list()
        for i in range(self.epoch):
            rand_index = np.random.choice(len(self.X), size=self.batch_size)
            rand_x = self.X[rand_index]
            rand_y = np.transpose([self.y[rand_index]])
            sess.run(self.optimize, feed_dict={self.data: rand_x, self.target: rand_y})
            rand_index = np.random.choice(len(self.X), size=self.batch_size)
            rand_x = self.X[rand_index]
            rand_y = np.transpose([self.y[rand_index]])
            test_accuracy = sess.run(self.accuracy, feed_dict={self.data: rand_x, self.target: rand_y})
            print("The accuracy at epoch",i,"=",test_accuracy)
