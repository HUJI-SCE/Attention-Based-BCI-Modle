import matplotlib.pyplot as plt
from matplotlib import use
from utils import get_pickle


def plot_loss(state_model_index):
    """
    This function plots the loss as a function of the interation index
    as the model is being trained.

    :param: state_model_index - the model index which iterative loss should be plotted
    """
    # set index path
    index_loss_path = ['model_states', f'model_{state_model_index}', 'loss.pickle']

    # get loss data
    loss_data = get_pickle(index_loss_path)

    use('TkAgg')
    plt.plot(loss_data.detach().numpy())
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Loss Throughout Training')
    plt.show()


if __name__ == '__main__':
    plot_loss(0)
