import matplotlib.pyplot as plt


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

    # always disable interactive mode
    plt.ioff()

    # create the graph
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot
    ax.plot(indexes, train, 'b-', label='Train set', linewidth=2.0)
    ax.plot(indexes, test, 'r--', label='Test set', linewidth=2.0)

    # visual settings
    ax.set_title('Accuracy of the ' + name, fontsize=16, y=1.02)
    ax.legend(loc='upper left', shadow=True)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim((0, 1))
    ax.grid(True)

    # save
    fig.savefig(path + '.png')
