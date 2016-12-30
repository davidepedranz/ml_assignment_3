from collections import namedtuple
import tensorflow as tf
import numpy as np
import sys
import string


def training_parameters():
    Params = namedtuple('Params', ['epochs', 'batch_size', 'summary_every', 'logs_path', 'num_sample_acc', 'seed'])
    name = string.replace(sys.modules['__main__'].__file__, '.py', '')
    return Params(epochs=20001, batch_size=50, summary_every=100, logs_path=name, num_sample_acc=1000, seed=1)


def mnist_subset(mnist, num_samples, seed):
    # force always the same seed
    np.random.seed(seed)

    # extract a subset of the train set
    train_subset = np.random.choice(mnist.train.num_examples, num_samples)
    train_images = mnist.train.images[train_subset]
    train_labels = mnist.train.labels[train_subset]

    # extract a subset of the test set
    test_subset = np.random.choice(mnist.test.num_examples, num_samples)
    test_images = mnist.test.images[test_subset]
    test_labels = mnist.test.labels[test_subset]

    return train_images, train_labels, test_images, test_labels


def bias_variable(shape):
    """
    Create a bias variable of the given shape which is initialized with a random value closed to zero.
    :param shape: Shape of the variable.
    :return:      Variable with the given shape, initialized with a value closed to zero.
    """
    with tf.name_scope('biases'):
        initial = tf.constant(0.1, shape=shape)
        variable = tf.Variable(initial)
    return variable


def weight_variable(shape):
    """
    Create a weights variable of the given shape which is initialized with a fixed value closed to zero.
    :param shape: Shape of the variable.
    :return:      Variable with the given shape, initialized with a value closed to zero.
    """
    with tf.name_scope('weights'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        variable = tf.Variable(initial)
    return variable


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def mnist_input_layer(name):
    """
    Create an input layer for the MNIST dataset.
    :param name: Namespace name for the new layer (used for the TensorBoard tool).
    :return: Pair of x_flatten and x_image
    """
    with tf.name_scope(name):
        x_flatten = tf.placeholder(tf.float32, shape=[None, 784], name='x_flatten')
        x_image = tf.reshape(x_flatten, [-1, 28, 28, 1], name='x_image')
    return x_flatten, x_image


def mnist_output_layer(name):
    """
    Create an output layer for the MNIST dataset.
    :param name: Namespace name for the new layer (used for the TensorBoard tool).
    :return: Pair of the predicted y and its label.
    """
    with tf.name_scope(name):
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
        y_label = tf.argmax(y_, 1, name='y_label')
    return y_, y_label


def convolutional_layer(input_tensor, input_dim, output_dim, name):
    """
    Create a new convolutional layer.
    :param input_tensor: Input of the new convolutional layer (must be a tensor).
    :param input_dim:    Dimension of the input features (must be a list).
    :param output_dim:   Dimension of the output features (must be a list).
    :param name:         Namespace name for the new layer (used for the TensorBoard tool).
    :return:             Activation value of the layer.
    """
    with tf.name_scope(name):
        weights = weight_variable(input_dim + output_dim)
        biases = bias_variable(output_dim)
        with tf.name_scope('convolution'):
            h_conv = tf.nn.relu(conv2d(input_tensor, weights) + biases)
    return h_conv


def pooling_layer(input_tensor, name):
    """
    Create a new convolutional layer.
    :param input_tensor: Input of the new polling layer (must be a tensor).
    :param name:         Namespace name for the new layer (used for the TensorBoard tool).
    :return:             Activation value of the layer.
    """
    with tf.name_scope(name):
        h_pool = max_pool_2x2(input_tensor)
    return h_pool


def connected_layer(input_tensor, input_dim, output_dim, name):
    """
    Create a fully connected layer of neurons with a ReLU activation function.
    :param input_tensor: Input of the new convolutional layer (must be a tensor).
    :param input_dim:    Dimension of the input features (must be a list).
    :param output_dim:   Dimension of the output features (must be a list).
    :param name:         Namespace name for the new layer (used for the TensorBoard tool).
    :return:             Activation value of the layer.
    """
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            weights = weight_variable(input_dim + output_dim)
        with tf.name_scope('biases'):
            biases = bias_variable(output_dim)
        with tf.name_scope('activation'):
            activation = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        return activation


def relu_dropout_layer(input_tensor, input_dim, output_dim, name):
    """
    Create a fully connected layer of neurons with a ReLU activation function.
    :param input_tensor: Input of the new convolutional layer (must be a tensor).
    :param input_dim:    Dimension of the input features (must be a list).
    :param output_dim:   Dimension of the output features (must be a list).
    :param name:         Namespace name for the new layer (used for the TensorBoard tool).
    :return:             Activation value of the layer and probability variable for Dropout.
    """
    with tf.name_scope(name):
        weights = weight_variable(input_dim + output_dim)
        biases = bias_variable(output_dim)
        with tf.name_scope('ReLU'):
            relu_activation = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            dropout_activation = tf.nn.dropout(relu_activation, keep_prob)
    return dropout_activation, keep_prob


def softmax_layer(input_tensor, input_dim, output_dim, name):
    """
    Create a new Softmax layer.
    :param input_tensor: Input of the new convolutional layer (must be a tensor).
    :param input_dim:    Dimension of the input features (must be a list).
    :param output_dim:   Dimension of the output features (must be a list).
    :param name:         Namespace name for the new layer (used for the TensorBoard tool).
    :return:             Activation value of the layer and computed label.
    """
    with tf.name_scope(name):
        weights = weight_variable(input_dim + output_dim)
        biases = bias_variable(output_dim)
        with tf.name_scope('activation'):
            activation = tf.matmul(input_tensor, weights) + biases
        with tf.name_scope('label'):
            label = tf.argmax(activation, 1)
        return activation, label


def accuracy_fun(prediction, real):
    """
    Compute the cross entropy function of a scalar prediction.
    :param prediction: Predicted value (scalar label).
    :param real:       Real value (scalar label).
    :return:           TensorFlow task to compute the accuracy metric.
    """
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(prediction, real)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


def cross_entropy_fun(prediction, real):
    """
    Compute the cross entropy function of an one-hot-encoded prediction.
    :param prediction: Predicted value (one-hot-encoded array).
    :param real:       Real value (one-hot-encoded array).
    :return:           TensorFlow task to compute the cross entropy.
    """
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, real))
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy
