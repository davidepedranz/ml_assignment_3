from export_utitilies import mkdir
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def smooth(x, y):
    smoother = interp1d(x, y, kind='cubic')
    x_smooth = np.linspace(x.min(), x.max(), num=50, endpoint=True)
    y_smooth = smoother(x_smooth)
    return x_smooth, y_smooth


def main():
    # compute real path
    mkdir('graphs')

    # load data
    epoch_comp, _, test_acc_comp = np.loadtxt('csv/partial/model_complete.csv', delimiter=",", skiprows=1, unpack=True)
    epoch_1, _, test_acc_1 = np.loadtxt('csv/partial/model_no1.csv', delimiter=",", skiprows=1, unpack=True)
    epoch_2, _, test_acc_2 = np.loadtxt('csv/partial/model_no2.csv', delimiter=",", skiprows=1, unpack=True)
    epoch_3, _, test_acc_3 = np.loadtxt('csv/partial/model_no3.csv', delimiter=",", skiprows=1, unpack=True)

    # smooth data
    epoch_comp_smooth, test_acc_comp_smooth = smooth(epoch_comp, test_acc_comp)
    epoch_1_smooth, test_acc_1_smooth = smooth(epoch_1, test_acc_1)
    epoch_2_smooth, test_acc_2_smooth = smooth(epoch_2, test_acc_2)
    epoch_3_smooth, test_acc_3_smooth = smooth(epoch_3, test_acc_3)

    # always disable interactive mode
    plt.ioff()

    # create normal plot
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(epoch_comp_smooth, test_acc_comp_smooth, '-', label='Original Network', linewidth=2.0)
    ax.plot(epoch_1_smooth, test_acc_1_smooth, '--', label='Network 1', linewidth=2.0)
    ax.plot(epoch_2_smooth, test_acc_2_smooth, '-.', label='Network 2', linewidth=2.0)
    ax.plot(epoch_3_smooth, test_acc_3_smooth, ':', label='Network 3', linewidth=2.0)
    ax.set_title('Performances of the different networks', fontsize=16, y=1.02)
    ax.legend(loc='lower right', shadow=True)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy on the test set')
    ax.set_ylim((0.97, 1))
    ax.grid(True)
    fig1.savefig('graphs/performances.png')


# entry point
if __name__ == '__main__':
    main()
