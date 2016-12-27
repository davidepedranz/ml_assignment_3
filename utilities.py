import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_convolutional_layer(input_tensor, weights_dim, bias_dim, layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable(weights_dim)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable(bias_dim)
            variable_summaries(biases)
        with tf.name_scope('h_convolution'):
            h_conv = tf.nn.relu(conv2d(input_tensor, weights) + biases)
            tf.summary.histogram('h_conv', h_conv)
        with tf.name_scope('h_pooling'):
            h_pool = max_pool_2x2(h_conv)
            tf.summary.histogram('h_pool', h_pool)
        return h_pool


def nn_densely_connected_layer(input_tensor, input_dim, output_dim, layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('activation'):
            activation = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
            tf.summary.histogram('activation', activation)
        return activation


def nn_softmax(input_tensor, input_dim, output_dim, layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('activation'):
            activation = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('activation', activation)
        return activation


def cross_entropy_function(prediction, real):
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, real))
    return cross_entropy


def accuracy_function(prediction, real):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(real, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
