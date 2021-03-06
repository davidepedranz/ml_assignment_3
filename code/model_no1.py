from tensorflow.contrib.learn.python.learn import datasets
from nn_utilities import *
from export_utitilies import save_csv


def main():
    """
    Like "model_complete.py", but without the first layer of convolution and pooling.
    """

    # force the same seed
    np.random.seed(get_seed())
    tf.set_random_seed(get_seed())

    # training details
    params = training_parameters()

    # import the MNIST dataset
    mnist = datasets.mnist.read_data_sets('data', one_hot=True)

    # input
    x, x_image = mnist_input_layer(name='input')

    # 1st layer: removed!
    # 2nd layer: Second Convolutional + Pooling Layer
    layer_2_conv = convolutional_layer(x_image, [5, 5, 1], [64], name='convolution_2')
    layer_2_pool = pooling_layer(layer_2_conv, name='pooling_2')

    # flatten the input, so that it can fit the 3rd layer
    with tf.name_scope('flatten'):
        layer_2_flat = tf.reshape(layer_2_pool, [-1, 14 * 14 * 64])

    # 3rd layer: Densely Connected Layer + Dropout
    layer_3, keep_prob = relu_dropout_layer(layer_2_flat, [14 * 14 * 64], [1024], name='relu_dropout_3')

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

        # save the result of accuracy to plot them
        accuracies_indexes = []
        accuracies_train = []
        accuracies_test = []

        # save the result of accuracy on the whole dataset
        accuracies_whole_indexes = []
        accuracies_whole_train = []
        accuracies_whole_test = []

        # NB: since my machine is slow, measure the performances during training
        #     using only a random selection of samples
        train_images, train_labels, test_images, test_labels = mnist_subset(mnist, params.num_sample_acc)

        # initialize the variables
        sess.run(init)

        # train the network using batches of training examples
        print('\nStart of training:')
        for i in range(1, params.epochs + 1):
            xs, ys = mnist.train.next_batch(params.batch_size)

            # train the network
            sess.run(train_step, feed_dict={x: xs, y_: ys, keep_prob: 0.5})

            # every x epochs, register train and test accuracy
            if i % params.summary_every == 0:
                # save accuracy for plot
                train_acc = sess.run(accuracy, feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0})
                test_acc = sess.run(accuracy, feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})
                accuracies_indexes.append(i)
                accuracies_train.append(train_acc)
                accuracies_test.append(test_acc)

                # print the results
                print(' * epoch %5d of %d ... train accuracy = %0.5f, test accuracy = %0.5f' %
                      (i, params.epochs, train_acc, test_acc))

            # every y epochs, register total train and test accuracy
            if i % params.complete_summary_every == 0:
                train_acc = sess.run(accuracy, feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0})
                test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                accuracies_whole_indexes.append(i)
                accuracies_whole_train.append(train_acc)
                accuracies_whole_test.append(test_acc)

        # print the results at the end of the training
        print('\nEnd of training... checkout the logs files!\n')

    # end of training session
    sess.close()

    # save data as CSV
    save_csv(accuracies_indexes, accuracies_train, accuracies_test, params.logs_path, 'partial')
    save_csv(accuracies_whole_indexes, accuracies_whole_train, accuracies_whole_test, params.logs_path, 'whole')


# entry point
if __name__ == '__main__':
    main()
