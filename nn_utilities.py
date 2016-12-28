import tensorflow as tf


def initialize_variable(shape):
    """
    Create a variable of the given shape which is initialized with a random value closed to zero.
    :param shape: Shape of the variable.
    :return:      Variable with the given shape, initialized with a value closed to zero.
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def convolutional_layer(input_tensor, input_dim, output_dim, layer_name):
    """
    Create a new convolutional layer.
    :param input_tensor: Input of the new convolutional layer (must be a tensor).
    :param input_dim:    Dimension of the input features (must be a list).
    :param output_dim:   Dimension of the output features (must be a list).
    :param layer_name:   Namespace name for the new layer (used for the TensorBoard tool).
    :return:             Activation value of the layer.
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = initialize_variable(input_dim + output_dim)
        with tf.name_scope('biases'):
            biases = initialize_variable(output_dim)
        with tf.name_scope('h_convolution'):
            h_conv = tf.nn.relu(tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding='SAME') + biases)
        with tf.name_scope('h_pooling'):
            h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return h_pool


def connected_layer(input_tensor, input_dim, output_dim, layer_name):
    """
    Create a fully connected layer of neurons with a ReLU activation function.
    :param input_tensor: Input of the new convolutional layer (must be a tensor).
    :param input_dim:    Dimension of the input features (must be a list).
    :param output_dim:   Dimension of the output features (must be a list).
    :param layer_name:   Namespace name for the new layer (used for the TensorBoard tool).
    :return:             Activation value of the layer.
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = initialize_variable(input_dim + output_dim)
        with tf.name_scope('biases'):
            biases = initialize_variable(output_dim)
        with tf.name_scope('activation'):
            activation = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        return activation


def softmax_layer(input_tensor, input_dim, output_dim, layer_name):
    """
    Create a new Softmax layer.
    :param input_tensor: Input of the new convolutional layer (must be a tensor).
    :param input_dim:    Dimension of the input features (must be a list).
    :param output_dim:   Dimension of the output features (must be a list).
    :param layer_name:   Namespace name for the new layer (used for the TensorBoard tool).
    :return:             Activation value of the layer.
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = initialize_variable(input_dim + output_dim)
        with tf.name_scope('biases'):
            biases = initialize_variable(output_dim)
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
    return cross_entropy
