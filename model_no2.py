import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import datasets
from nn_utilities import convolutional_layer, connected_layer, softmax_layer, cross_entropy_fun, accuracy_fun
from export_utitilies import plot, save_csv


def main():
    """
    Train a Deep Network with 1 convolutional layer, a fully connected layer
    with dropout and an output softmax layer for the MNIST dataset.
    """

    # training details
    N_EPOCHS = 10001
    SUMMARY_EVERY = 100
    BATCH_SIZE = 50
    NETWORK_NAME = 'model 2'
    PATH = '02_model_2'

    # import the MNIST dataset
    mnist = datasets.mnist.read_data_sets('data', one_hot=True)

    # input
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x_flatten')
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

    # only 1st Convolutional Layer
    layer_1 = convolutional_layer(x_image, [5, 5, 1], [16], '1_convolutional')
    layer_1_flat = tf.reshape(layer_1, [-1, 14 * 14 * 16], 'flatten')

    # 3rd layer: Densely Connected Layer + Dropout
    layer_3 = connected_layer(layer_1_flat, [14 * 14 * 16], [1024], '3_connected')
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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
        train_subset = np.random.choice(mnist.train.num_examples, mnist.train.num_examples / 11)
        train_images = mnist.train.images[train_subset]
        train_labels = mnist.train.labels[train_subset]

        # initialize the variables
        sess.run(init)

        # train the network using batches of training examples
        print('\nStart of training:')
        for i in range(1, N_EPOCHS + 1):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            # train the network
            sess.run(train_step, feed_dict={x: xs, y_: ys, keep_prob: 0.5})

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

        # print the results at the end of the training
        train_acc = sess.run(accuracy, feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0})
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print('\nEnd of training:')
        print(' * train accuracy = %g' % train_acc)
        print(' * test accuracy  = %g' % test_acc)

    # end of training session
    sess.close()

    # plot the graph & save data as CSV
    plot(accuracies_indexes, accuracies_train, accuracies_test, NETWORK_NAME, PATH)
    save_csv(accuracies_indexes, accuracies_train, accuracies_test, PATH)


# entry point
if __name__ == '__main__':
    main()
