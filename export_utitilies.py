import errno
import os
import numpy as np


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
