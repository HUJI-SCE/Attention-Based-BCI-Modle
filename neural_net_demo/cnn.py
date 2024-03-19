import os
from datetime import date

import numpy as np
import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
import mne
import pickle
import scipy
import torch.optim as optim

# imported local files from project
from preprocess import preprocess

# More or less => ssh -l loi201loi phoenix.cs.huji.ac.il
# nqouta    usage of disk space
# ssinfo    jobs that are running now
# ssqueue   queues of these jobs
# susage    Recap per user jobs that are running now

# starting new sessions that won't close upon disconnection:
# tmux to start a new session
# tmux -a to open existing session

# Set up input data
total_samples = 3_072_000
electrode_count = 144
samples_per_batch = 77
kernel_width = 10
kernel_stride = 1
fc_segment_length = 64

temp_data_set = torch.randn(electrode_count, samples_per_batch)  # temp random data
data_labels = torch.from_numpy(np.random.choice([0, 1], size=(1,), p=[1. / 3, 2. / 3])).type(torch.FloatTensor)
# [expression for item in iterable if condition == True]
temp_many_data_sets =  [torch.randn(electrode_count, samples_per_batch) for i in range(10)]


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
            topology_metadata = [(144, 77), (10, 1, 64)]

        # decomposition of the metadata variables in topology
        channel_count, batch_width = topology_metadata[0]
        kernel_width, kernel_stride, fc_segment_length = topology_metadata[1]

        second_layer_out_channels = 128
        third_layer_out_channels = 1

        # list of linear transformations for the segments in the 1st layer.
        self.fc_layers = nn.ModuleList(
            [nn.Linear(batch_width, fc_segment_length) for _ in range(channel_count)])

        # We have a kernel, Matrix[144][10]. We convolute this with the Batch (Matrix[144][77])
        # We run this Matrix right\left over the batch (thus 67 output_channels in the end because stride is 1)
        # a single transformation for the second part of the 1st layer
        self.conv_segment = CustomConvSegment(channel_count, kernel_width, kernel_stride)

        # 2nd layer
        self.fast_reduction_layer = nn.Linear(channel_count * fc_segment_length +
                                              (batch_width - kernel_width + 1),
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


# First run preproccess.py (as main) to create updated pickle files from kayas data
# Call a function from preproccess to Load the pickle data, it returns numpy array.

def create_model():
    # Instantiate your model
    model = MyModel(topology_metadata=[(electrode_count, samples_per_batch),
                                       (kernel_width, kernel_stride, fc_segment_length)])
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


def save_model(model):
    model_dir = os.getcwd()
    model_path = os.path.join(model_dir, "model" + ".pth")
    torch.save(model, model_path)


def train_model(model, data_set):
    """First we are moving our data to GPU and then removing any gradient
     so that it doesn’t effect our training.
      Next we are making a forward pass and obtain the outputs.
      Next we pass these output along with correct label to obtain our loss
       and then call backward() to calculate gradients.
        Then we call step() to update the weights. And add the loss to train_loss."""

    is_gpu = torch.cuda.is_available()
    is_gpu = False

    epochs = 5
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.7)

    for i in range(1):
        train_loss = 0.0
        # for data, label in (temp_many_data_sets, data_labels):
        for j in range(1):
            data, label = data_set, data_labels
            if is_gpu:
                data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, label)
            loss.backward()

            optimizer.step()

            train_loss += loss.item() * data.size(0)
        print(f'Epoch: {i + 1} / {epochs} \t\t\t Training Loss:{train_loss / len(data_set)}')
    save_model(model)


def main():
    # Loading model from path:
    #model = torch.load(PATH)
    model = create_model()
    train_model(model, temp_data_set)
    # profile_model(model)


if __name__ == '__main__':
    main()



"""
This is very well documented on the PyTorch website,
you definitely won't regret spending a minute or two reading this page. 
PyTorch essentially defines nine CPU tensor types and nine GPU tensor types:

╔══════════════════════════╦═══════════════════════════════╦════════════════════╦═════════════════════════╗
║        Data type         ║             dtype             ║     CPU tensor     ║       GPU tensor        ║
╠══════════════════════════╬═══════════════════════════════╬════════════════════╬═════════════════════════╣
║ 32-bit floating point    ║ torch.float32 or torch.float  ║ torch.FloatTensor  ║ torch.cuda.FloatTensor  ║
║ 64-bit floating point    ║ torch.float64 or torch.double ║ torch.DoubleTensor ║ torch.cuda.DoubleTensor ║
║ 16-bit floating point    ║ torch.float16 or torch.half   ║ torch.HalfTensor   ║ torch.cuda.HalfTensor   ║
║ 8-bit integer (unsigned) ║ torch.uint8                   ║ torch.ByteTensor   ║ torch.cuda.ByteTensor   ║
║ 8-bit integer (signed)   ║ torch.int8                    ║ torch.CharTensor   ║ torch.cuda.CharTensor   ║
║ 16-bit integer (signed)  ║ torch.int16 or torch.short    ║ torch.ShortTensor  ║ torch.cuda.ShortTensor  ║
║ 32-bit integer (signed)  ║ torch.int32 or torch.int      ║ torch.IntTensor    ║ torch.cuda.IntTensor    ║
║ 64-bit integer (signed)  ║ torch.int64 or torch.long     ║ torch.LongTensor   ║ torch.cuda.LongTensor   ║
║ Boolean                  ║ torch.bool                    ║ torch.BoolTensor   ║ torch.cuda.BoolTensor   ║
╚══════════════════════════╩═══════════════════════════════╩════════════════════╩═════════════════════════╝
"""