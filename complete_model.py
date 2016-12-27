from tensorflow.examples.tutorials.mnist import input_data
from utilities import *


def main():
    ########################################
    # import the MNIST dataset
    ########################################
    mnist = input_data.read_data_sets('data', one_hot=True)

    ########################################
    # define the network
    ########################################

    # input
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x_flatten_1D')
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image_2D')

    # 1st layer: First Convolutional Layer
    layer_1 = nn_convolutional_layer(x_image, [5, 5, 1, 32], [32], '1st_convolutional_layer')

    # 2nd layer: Second Convolutional Layer
    layer_2 = nn_convolutional_layer(layer_1, [5, 5, 32, 64], [64], '2nd_convolutional_layer')
    layer_2_flat = tf.reshape(layer_2, [-1, 7 * 7 * 64], 'flatten_layer_2')

    # 3rd layer: Densely Connected Layer + Dropout
    layer_3 = nn_densely_connected_layer(layer_2_flat, 7 * 7 * 64, 1024, 'densely_connected_and_dropout')
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        layer_3_dropped = tf.nn.dropout(layer_3, keep_prob)

    # 4th layer: Readout Layer
    y_conv = nn_softmax(layer_3_dropped, 1024, 10, 'softmax_layer')

    # output
    with tf.name_scope('output'):
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_one_hot_encoded')

    # training step
    cross_entropy = cross_entropy_function(y_conv, y_)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # evaluation metrics
    accuracy = accuracy_function(y_conv, y_)

    ########################################
    # train the network
    ########################################
    sess = tf.Session()
    with sess.as_default():

        # initialize the variables
        sess.run(tf.global_variables_initializer())

        # train the network using batches of training examples
        for i in range(6):
            batch = mnist.train.next_batch(50)
            if i % 5 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % sess.run(accuracy, feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    sess.close()

    ########################################
    # save the network
    ########################################
    tf.summary.FileWriter('logs', sess.graph)


# entry point
if __name__ == '__main__':
    main()
