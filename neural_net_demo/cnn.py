import os

import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
import torch.optim as optim

# imported local files from project
from preprocess import *
from state_model_index import StateModelIndex


# Set up input data
total_samples = 3_072_000
electrode_count = 144
samples_per_batch = 77
kernel_width = 10
kernel_stride = 1
fc_segment_length = 64

model_dir = 'model_states'

# temp_data_set = torch.randn(electrode_count, samples_per_batch)  # temp random data
# data_labels = torch.from_numpy(np.random.choice([0, 1], size=(1,), p=[1. / 3, 2. / 3])).type(torch.FloatTensor)
# [expression for item in iterable if condition == True]
# temp_many_data_sets = [torch.randn(electrode_count, samples_per_batch) for i in range(10)]


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

        * topology: [(channel_count, batch_width), (kernel_width, kernel_stride, fc_segment_length)]
        """

        super(MyModel, self).__init__()

        if topology_metadata is None:
            topology_metadata = [(144, 77), (10, 1, 64), (128, 1)]

        # decomposition of the metadata variables in topology
        channel_count, batch_width = topology_metadata[0]
        kernel_width, kernel_stride, fc_segment_length = topology_metadata[1]
        second_layer_out_channels, third_layer_out_channels = topology_metadata[2]

        # list of linear transformations for the segments in the 1st layer.
        self.fc_layers = nn.ModuleList(
            [nn.Linear(batch_width, fc_segment_length) for _ in range(channel_count)])

        # We have a kernel, Matrix[144][10]. We convolve this with the Batch (Matrix[144][77])
        # We run this Matrix right\left over the batch (thus 67 output_channels in the end because stride is 1)
        # a single transformation for the second part of the 1st layer
        self.conv_segment = CustomConvSegment(channel_count, kernel_width, kernel_stride)

        # 2nd layer: segment_2a/2b (see documentation).
        segment_2a_length = (batch_width - kernel_width + 1)
        segment_2b_length = channel_count * fc_segment_length

        self.fast_reduction_layer = nn.Linear(segment_2b_length + segment_2b_length,
                                              second_layer_out_channels)
        # 3rd layer
        self.reduction_to_output_layer = nn.Linear(second_layer_out_channels, third_layer_out_channels)

        # activation function embedded in all layers
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Fully connected layers for each channel
        # list of applying the random Linear transformations on each electrode (list of segments [exciting!])
        fc_outputs = [fc(x[i, :]) for i, fc in enumerate(self.fc_layers)]
        # convert to Tensor and not List of Tensors:
        fc_outputs = torch.cat(fc_outputs)

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
        x = self.sigmoid(x)

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


# First run preprocess.py (as main) to create updated pickle files from kayas data
# Call a function from preprocess to Load the pickle data, it returns numpy array.

def create_model():
    # Instantiate your model
    model = MyModel()
    return model


def profile_model(model, data_set):
    print(model)
    # Run the forward pass with profiling
    with profiler.profile(record_shapes=True, use_cuda=False) as prof:
        with profiler.record_function("forward_pass"):
            output = model(data_set)

    # Print the profiling results
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))


def timestamp():
    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    # dd_mm_YY-H:M:S
    dt_string = now.strftime("%d_%m_%Y-%H:%M:%S")
    return dt_string


def save_model(model: MyModel, shuffle: np.ndarray, loss, state_model_index=0):
    """
    This function stores the model state (all current weights and biases) in an indexed file,
    after training on a mini-batch (an ordered list of input vectors,
    which induce a series of forward passes).

    It also stores the shuffle of the mini-batch - the order of the input vectors in the mini-batch.
    This is the order in which the input vectors, that are originally ordered sequentially
    with respect to the recording (of the EEG), should be fed into the network (This value
    is randomly generated).

    todo: determine whether the shuffle should be generated at this stage, or at another stage
        and implement.

    :param: model - the current state of the model, as a pytorch network object.
    :param: shuffle - the order in which the input vectors.
    :param: the training loss of the model
    :state_index: the model_state_index (see StateModelIndex documentation)
    """

    # save model state in suitable file
    os.system(os.path.join(model_dir, f'model_{state_model_index}'))
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model, model_path)

    # document the shuffle
    with open(os.path.join(model_dir, 'shuffle.pickle'), 'wb') as file:
        pickle.dump(shuffle, file)

    # document the details: timestamp, loss...(TBD)
    with open(os.path.join(model_dir, 'details.txt'), 'w') as file:
        file.write(f'{timestamp()}\n')
        file.write(f'training loss: {loss}')


def train_model(model, data, label, shuffle, state_model_index):
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
    epochs = 5
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.7)
    train_loss = 0.0

    # train loop
    for data_vector, label_vector in zip(data[shuffle], label[shuffle]):
        optimizer.zero_grad()

        output = model(data_vector)
        loss = criterion(output, label_vector)
        loss.backward()

        optimizer.step()

        train_loss += loss.item() * data.size(0)
    # print(f'Epoch: {i + 1} / {epochs} \t\t\t Training Loss:{train_loss / len(data_set)}')
    save_model(model, shuffle, train_loss, state_model_index)


def main():

    model = create_model()

    # init the model state index to it's last value
    with StateModelIndex('state_model_index') as state_model_index:

        # Loading model from path (if the path exists)
        try:
            model = torch.load(os.path.join(model_dir, f'model_{state_model_index}.pth'))

        # Whether loading succeeded or failed -> start training session
        finally:
            # get dataset and shuffle from preprocess module
            (train_set, test_set, validate_set), \
                (train_solution, test_solution, validate_solution), \
                shuffle_train, shuffle_test, shuffle_validate = get_dataset()

            train_model(model, train_set, train_solution, shuffle_train, state_model_index)
    # profile_model(model)


if __name__ == '__main__':
    main()
