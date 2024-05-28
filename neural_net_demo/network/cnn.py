import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

# imported local files from project
from preprocess import Preprocessor
from state_model_index import StateModelIndex
from utils import store_pickle, timestamp

MODEL_PTH = 'model.pth'
MODEL_DIR = 'model_states'


class MyModel(nn.Module):
    def __init__(self, topology_metadata=None):
        """
        (see topology specification at the bottom (*) for variable terminology)
        This model is a simple CNN that is meant to be trained on EEG-data.

        The topology of the network is as follows:
        1. The 1st layer is the input layer (a batch) - which consists of channel_count channels,
        each containing batch_width samples.
        So in total the batch is a matrix of (channel_count x batch_width).

        2. The 2nd layer is 2-fold:
            a. A convolutional segment (of nodes) derived by multiplying a kernel of dimensions
            (channel_count, kernel_width) with the i-th iteration matrix (data[:, i:i + kernel_width])
             - in total (batch_width - kernel_width + 1) nodes.
            b. A series of full-connected sub-layers, each sublayer consists of fc_segment_length nodes
            that are part of the second layer.
            There is a one-to-one correspondence between the channels and the sub-layers.
            Each channel is fully connected to one sublayer.
            Each sublayer is equipped with the Relu activation function.
            This gives us another (channel_count * fc_segment_length) nodes in teh 2nd layer.

        3. The 3rd layer is a fully connected layer, converging the network to 128 nodes
        4. The 4th layer is a fully connected layer, converging the network to 1 output node.
        It is equipped with the Sigmoid activation functino.

        topology_metadata: a list of the number of nodes in every layer without any information about the topology
        The constructor decides what to do with the amounts given - thus, the caller needs to be
        mindful of the topology design in the constructor.

        * topology: [(channel_count, batch_width), (kernel_width, kernel_stride, fc_segment_length),
                     (second_layer_output, third_layer_output)]
        """

        super(MyModel, self).__init__()

        if topology_metadata is None:
            topology_metadata = [(22, 150), (10, 1, 64), (128, 9)]

        # decomposition of the metadata variables in topology
        channel_count, batch_width = topology_metadata[0]
        kernel_width, kernel_stride, fc_segment_length = topology_metadata[1]
        second_layer_out_channels, third_layer_out_channels = topology_metadata[2]

        # list of linear transformations for the segments in the 1st layer.
        self.fc_layers = nn.ModuleList(
            [nn.Linear(batch_width, fc_segment_length) for _ in range(channel_count)])

        # rearrange fc layers into tensor
        self.fc_tensor = torch.stack([fc.weight for fc in self.fc_layers]).permute(0, 2, 1)

        # We have a kernel, Matrix[22][10]. We convolve this with the Batch (Matrix[22][150])
        # We run this Matrix right\left over the batch (thus 67 output_channels in the end because stride is 1)
        # a single transformation for the second part of the 1st layer
        self.conv_segment = CustomConvSegment(channel_count, kernel_width, kernel_stride)

        # 2nd layer: segment_2a/2b (see __init__ documentation).
        segment_2a_length = (batch_width - kernel_width + 1)
        segment_2b_length = channel_count * fc_segment_length

        self.fast_reduction_layer = nn.Linear(segment_2a_length + segment_2b_length,
                                              second_layer_out_channels)
        # 3rd layer
        self.reduction_to_output_layer = nn.Linear(second_layer_out_channels, third_layer_out_channels)

        # activation function embedded in final layer alone
        # softmax: ensures that the result adds up to one - suitable for a distribution of a guess.
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # Fully connected layers for each channel
        fc_outputs = torch.matmul(x.unsqueeze(1), self.fc_tensor).flatten()

        # Convolutional segment (odd one out lol)
        conv_output = self.conv_segment(x)

        # merge the segments (fully connected from the electrodes),
        # with the convolutional segment to create the first layer.
        concatenated = torch.cat((fc_outputs, conv_output))
        # forward to 2nd layer (from concatenated layer)
        second_layer = nn.functional.relu(self.fast_reduction_layer(concatenated))
        # concatenated))

        # forward to 3rd (and output) layer (from 2nd layer)
        x = nn.functional.relu(self.reduction_to_output_layer(second_layer))

        # Apply sigmoid activation for binary classification
        x = self.softmax(x)

        return x


class CustomConvSegment(nn.Module):
    def __init__(self, channel_count, kernel_width, kernel_stride):
        super(CustomConvSegment, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=(channel_count, kernel_width),
                                    stride=kernel_stride)

    def forward(self, x):
        # Expand dimensions for the single-channel convolution, adds two dimensions, e.g: [1][1]...
        x = x.unsqueeze(0).unsqueeze(0)
        # Apply the custom convolutional layer
        x = self.conv_layer(x)
        # Flatten the output
        x = x.view(-1)
        return x


def create_model():
    # Instantiate your model
    model = MyModel()
    return model


def save_model(model: MyModel, shuffle: np.ndarray, loss_tracker, state_model_index=0):
    """
    This function stores the model state (all current weights and biases) in an indexed file,
    after training on a mini-batch (an ordered list of input vectors,
    which induce a series of forward passes).

    It also stores the shuffle of the mini-batch - the order of the input vectors in the mini-batch.
    This is the order in which the input vectors, that are originally ordered sequentially
    with respect to the recording (of the EEG), should be fed into the network (This value
    is randomly generated).

    :param: model - the current state of the model, as a pytorch network object.
    :param: shuffle - the order in which the input vectors.
    :param: the training loss of the model
    :state_index: the model_state_index (see StateModelIndex documentation)
    """

    # save model state in suitable file
    os.system(f'mkdir {os.path.join(MODEL_DIR, f"model_{state_model_index}")}')
    current_model_dir = os.path.join(MODEL_DIR, f'model_{state_model_index}')
    current_model_path = os.path.join(current_model_dir, 'model.pth')
    torch.save(model, current_model_path)

    # document the shuffle and the loss
    store_pickle(shuffle, [current_model_dir, 'shuffle.pickle'])
    store_pickle(loss_tracker, [current_model_dir, 'loss.pickle'])

    # document the details: timestamp, loss...(TBD)
    with open(os.path.join(current_model_dir, 'details.txt'), 'w') as file:
        file.write(f'{timestamp()}\n')
        file.write(f'training final loss: {loss_tracker[-1]}')


def train_model(model, data, label, shuffle, batch_size, state_model_index):
    """First we are moving our data to GPU and then removing any gradient
     so that it doesn't affect our training.
     Next we are making a forward pass and obtain the outputs.
     Next we pass these output along with correct label to obtain our loss
     and then call backward() to calculate gradients.
     Then we call step() to update the weights. And add the loss to train_loss."""

    # init gpu (if one exists)
    is_gpu = torch.cuda.is_available()
    if is_gpu:
        data, label = data.cuda(), label.cuda()

    # init training process parameters
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.7)
    train_loss = 0.0
    loss_tracker = torch.tensor([])

    # train loop
    print('starting time')
    t1 = time.time()
    for sh_index in shuffle[:1000]:
        optimizer.zero_grad()
        output = model(data[:, sh_index:sh_index + batch_size])

        loss = criterion(output, label[sh_index + batch_size])

        # save loss value, for analysis
        loss_tracker = torch.cat((loss_tracker, loss.view(1)), dim=0)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * data.size(0)
    print(time.time() - t1)
    # print(f'Epoch: {i + 1} / {epochs} \t\t\t Training Loss:{train_loss / len(data_set)}')
    save_model(model, shuffle[:1000], loss_tracker, state_model_index)


def main(data_object):

    model = create_model()

    # init the model state index to it's last value
    with StateModelIndex('state_model_index') as state_model_index:

        # Loading model from path (if the path exists)
        if state_model_index.value:
            print(os.getcwd())
            model = torch.load(os.path.join(MODEL_DIR, f'model_{state_model_index.value - 1}', MODEL_PTH))

        # get dataset and shuffle from preprocess module
        data_set = data_object.get_dataset()

        # convert float64 dataset from numpy array to float32 torch tensor
        label_set = data_set[1]
        data_set = torch.tensor(data_set[0], dtype=torch.float32)

        # train model
        train_model(model, data_set, label_set,
                    data_object.get_shuffle_set('TRAIN'), 150, state_model_index.value)

        # increment state model index
        state_model_index.set_smi(state_model_index.value + 1)

#with open('/cs/usr/loi201loi/Desktop/Project/neural_net_demo/out.txt', 'w') as f:
#    print('before if __name__',file=f)

if __name__ == '__main__':
    format_dictionary = {
        'data set key path': ['o', 0, 0, 5],
        'solution set key path': ['o', 0, 0, 4]
    }
    with open('/cs/usr/loi201loi/Desktop/Project/neural_net_demo/out.txt', 'w') as f:
        print('started if __name__',file=f)
        print('working dir is:', os.getcwd(), file=f)
    matlab_path = '../data/matlab_files/5F-SubjectH-160804-5St-SGLHand-HFREQ.mat'
    batch_parameters = (0.15, 1000)  # window interval in seconds, and frequency of EEG measurements
    train_test_validate_ratios = [0.6, 0.2, 0.2]
    with open('/cs/usr/loi201loi/Desktop/Project/neural_net_demo/out.txt', 'w') as f:
        print('We are entering preprocess', file=f)

    
    preprocessor = Preprocessor(format_dictionary, matlab_path, batch_parameters, train_test_validate_ratios)
    with open('/cs/usr/loi201loi/Desktop/Project/neural_net_demo/out.txt', 'w') as f:
        print('We are entering main', file=f)

    main(preprocessor)
