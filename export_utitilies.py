import errno
import os
import numpy as np
import matplotlib.pyplot as plt


def mkdir(path):
    """
    Creates a folder with the given path if it does not exist.
    :param path: Path.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_csv(accuracies_indexes, accuracies_train, accuracies_test, run_name):
    """
    Store the output of a run as a CSV file that can be used for further analysis.
    :param accuracies_indexes: Epochs.
    :param accuracies_train:   Train accuracy.
    :param accuracies_test:    Test accuracy.
    :param run_name:           Name of the run.
    """
    mkdir('csv')
    real_path = 'csv' + os.sep + run_name + '.csv'
    np.savetxt(real_path, zip(accuracies_indexes, accuracies_train, accuracies_test),
               header='epoch,accuracy_train,accuracy_test', delimiter=',', comments='')


def plot(indexes, train, test, name, path):
    """
    TODO
    :param indexes:
    :param train:
    :param test:
    :param name:
    :param path:
    :return:
    """

    # compute real path
    mkdir('graphs')
    real_path = 'graphs' + os.sep + path

    # always disable interactive mode
    plt.ioff()

    # create normal plot
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(indexes, train, 'b-', label='Train set', linewidth=2.0)
    ax.plot(indexes, test, 'r--', label='Test set', linewidth=2.0)
    ax.set_title('Accuracy of the ' + name, fontsize=16, y=1.02)
    ax.legend(loc='lower right', shadow=True)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim((0.8, 1))
    ax.grid(True)
    fig1.savefig(real_path + '.png')

    # create semilogy plot
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.semilogy(indexes, train, 'b-', label='Train set', linewidth=2.0)
    ax.semilogy(indexes, test, 'r--', label='Test set', linewidth=2.0)
    ax.set_title('Accuracy of the ' + name, fontsize=16, y=1.02)
    ax.legend(loc='lower right', shadow=True)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim((0.8, 1))
    ax.grid(True)
    fig1.savefig(real_path + '_semilogy.png')
