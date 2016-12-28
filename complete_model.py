import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import datasets
from nn_utilities import convolutional_layer, connected_layer, softmax_layer, cross_entropy_fun, accuracy_fun
from plot_utitilies import plot


def main():
    """
    Train a Deep Network with 2 convolutional layer, a fully connected layer
    with dropout and an output softmax layer for the MNIST dataset.
    """

    # training details
    N_EPOCHS = 20001
    SUMMARY_EVERY = 100
    BATCH_SIZE = 50
    NETWORK_NAME = 'original network'
    PATH = '00_original'

    # N_EPOCHS = 2
    # SUMMARY_EVERY = 1
    # BATCH_SIZE = 5

    # import the MNIST dataset
    mnist = datasets.mnist.read_data_sets('data', one_hot=True)

    # input
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x_flatten')
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

    # 1st layer: First Convolutional Layer
    layer_1 = convolutional_layer(x_image, [5, 5, 1], [32], '1_convolutional')

    # 2nd layer: Second Convolutional Layer
    layer_2 = convolutional_layer(layer_1, [5, 5, 32], [64], '2_convolutional')
    layer_2_flat = tf.reshape(layer_2, [-1, 7 * 7 * 64], 'flatten')

    # 3rd layer: Densely Connected Layer + Dropout
    layer_3 = connected_layer(layer_2_flat, [7 * 7 * 64], [1024], '3_connected')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    layer_3_dropped = tf.nn.dropout(layer_3, keep_prob)

    # 4th layer: Readout Layer
    y_predicted_one_hot, y_predicted_label = softmax_layer(layer_3_dropped, [1024], [10], '4_softmax')

    # output
    with tf.name_scope('real_output'):
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
        y_real_label = tf.argmax(y_, 1, name='y_label')

    # evaluation metrics
    accuracy = accuracy_fun(y_predicted_label, y_real_label)
    tf.summary.scalar('accuracy', accuracy)

    # training step
    cross_entropy = cross_entropy_fun(y_predicted_one_hot, y_)
    tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # initialization step
    with tf.name_scope('init'):
        init = tf.global_variables_initializer()

    # train the network
    sess = tf.Session()
    with sess.as_default():

        # register metrics
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('logs/' + PATH + '/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/' + PATH + '/test')

        # save the result of accuracy to plot them
        accuracies_indexes = []
        accuracies_train = []
        accuracies_test = []

        # NB: since my machine has not enough memory, measure the performances
        #     on the train set using a random selection of samples
        train_subset = np.random.choice(mnist.train.num_examples, mnist.train.num_examples / 5)
        train_images = mnist.train.images[train_subset]
        train_labels = mnist.train.labels[train_subset]

        # initialize the variables
        sess.run(init)

        # train the network using batches of training examples
        print('\nStart of training:')
        for i in range(0, N_EPOCHS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            # every x epochs, register train and test accuracy
            if i % SUMMARY_EVERY == 0:
                print(' * epoch %5d of %d ...' % (i, N_EPOCHS))

                # save metrics for TensorBoard
                summary_train = sess.run(merged, feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0})
                train_writer.add_summary(summary_train, i)
                summary_test = sess.run(merged, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                test_writer.add_summary(summary_test, i)

                # save accuracy for plot
                train_acc = sess.run(accuracy, feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0})
                test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                accuracies_indexes.append(i)
                accuracies_train.append(train_acc)
                accuracies_test.append(test_acc)

            # train the network
            sess.run(train_step, feed_dict={x: xs, y_: ys, keep_prob: 0.5})

        # print the results at the end of the training
        train_acc = sess.run(accuracy, feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0})
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print('\nEnd of training:')
        print(' * train accuracy = %g' % train_acc)
        print(' * test accuracy  = %g' % test_acc)

    # end of training session
    sess.close()

    # plot the graph
    plot(accuracies_indexes, accuracies_train, accuracies_test, NETWORK_NAME, PATH)


# entry point
if __name__ == '__main__':
    main()
