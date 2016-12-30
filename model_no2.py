from tensorflow.contrib.learn.python.learn import datasets
from nn_utilities import *
from export_utitilies import save_csv


def main():
    """
    Like Model Complete, but without the second layer.
    """

    # training details
    params = training_parameters()

    # import the MNIST dataset
    mnist = datasets.mnist.read_data_sets('data', one_hot=True)

    # force the seed
    tf.set_random_seed(params.seed)

    # input
    x_flatten, x_image = mnist_input_layer(name='input')

    # 1st layer: First Convolutional + Pooling Layer
    layer_1_conv = convolutional_layer(x_image, [5, 5, 1], [32], name='convolution_1')
    layer_1_pool = pooling_layer(layer_1_conv, name='pooling_1')

    # 2nd layer: dropped!

    # flatten the input, so that it can fit the 3rd layer
    with tf.name_scope('flatten'):
        layer_1_flat = tf.reshape(layer_1_pool, [-1, 14 * 14 * 32])

    # 3rd layer: Densely Connected Layer + Dropout
    layer_3, keep_prob = relu_dropout_layer(layer_1_flat, [14 * 14 * 32], [1024], name='relu_dropout_3')

    # 4th layer: Readout Layer
    y_predicted_one_hot, y_predicted_label = softmax_layer(layer_3, [1024], [10], name='softmax_4')

    # output
    y_, y_label = mnist_output_layer(name='output')

    # evaluation metrics
    accuracy = accuracy_fun(y_predicted_label, y_label)

    # training step
    cross_entropy = cross_entropy_fun(y_predicted_one_hot, y_)
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
        train_writer = tf.summary.FileWriter('logs/' + params.logs_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/' + params.logs_path + '/test')

        # save the result of accuracy to plot them
        accuracies_indexes = []
        accuracies_train = []
        accuracies_test = []

        # NB: since my machine has not enough memory, measure the performances
        #     on the train set using a random selection of samples
        train_images, train_labels, test_images, test_labels = mnist_subset(mnist, params.num_sample_acc, params.seed)

        # initialize the variables
        sess.run(init)

        # train the network using batches of training examples
        print('\nStart of training:')
        for i in range(1, params.epochs + 1):
            xs, ys = mnist.train.next_batch(params.batch_size)

            # train the network
            sess.run(train_step, feed_dict={x_flatten: xs, y_: ys, keep_prob: 0.5})

            # every x epochs, register train and test accuracy
            if i % params.summary_every == 0:
                # save metrics for TensorBoard
                summary_train = sess.run(merged, feed_dict={x_flatten: train_images, y_: train_labels, keep_prob: 1.0})
                train_writer.add_summary(summary_train, i)
                summary_test = sess.run(merged, feed_dict={x_flatten: test_images, y_: test_labels, keep_prob: 1.0})
                test_writer.add_summary(summary_test, i)

                # save accuracy for plot
                train_acc = sess.run(accuracy, feed_dict={x_flatten: train_images, y_: train_labels, keep_prob: 1.0})
                test_acc = sess.run(accuracy, feed_dict={x_flatten: test_images, y_: test_labels, keep_prob: 1.0})
                accuracies_indexes.append(i)
                accuracies_train.append(train_acc)
                accuracies_test.append(test_acc)

                # print the results
                print(' * epoch %5d of %d ... train accuracy = %0.5f, test accuracy = %0.5f' %
                      (i, params.epochs, train_acc, test_acc))

        # print the results at the end of the training
        train_acc = sess.run(accuracy, feed_dict={x_flatten: train_images, y_: train_labels, keep_prob: 1.0})
        test_acc = sess.run(accuracy, feed_dict={x_flatten: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print('\nEnd of training:')
        print(' * total train accuracy = %g' % train_acc)
        print(' * total test accuracy  = %g' % test_acc)

    # end of training session
    sess.close()

    # save data as CSV
    save_csv(accuracies_indexes, accuracies_train, accuracies_test, params.logs_path)


# entry point
if __name__ == '__main__':
    main()
