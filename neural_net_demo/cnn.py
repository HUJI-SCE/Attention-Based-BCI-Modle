import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
import mne
import pickle

# imported local files from project
from preprocess import preprocess


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

        second_layer_out_channels   = 128
        third_layer_out_channels    = 1

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
            #concatenated))

        # forward to 3rd (and output) layer (from 2nd layer)
        x = nn.functional.relu(self.reduction_to_output_layer(second_layer))

        # Apply sigmoid activation for binary classification
        x = self.sigmoid(x)

        return x


class CustomConvSegment(nn.Module):
    def __init__(self, channel_count, kernel_width, kernel_stride):
        super(CustomConvSegment, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels =1,
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


# Set up input data
total_samples       = 3_072_000
electrode_count     = 144
samples_per_batch   = 77
kernel_width        = 10
kernel_stride       = 1
fc_segment_length   = 64


#raw = mne.io.read_raw_fif('raw_complete.fif', preload=True)
# unpickle data
#with open('numpy_data.pickle', 'rb') as raw_pickle:
#    data_set = pickle.load(raw_pickle)
#batches = preprocess(data_set)


data_set = torch.randn(electrode_count, samples_per_batch) #temp random data
# Instantiate your model
model = MyModel(topology_metadata=[(electrode_count, samples_per_batch),
                                   (kernel_width,  kernel_stride, fc_segment_length)])

print(data_set)
print(model)
# Run the forward pass with profiling
with profiler.profile(record_shapes=True, use_cuda=False) as prof:
    with profiler.record_function("forward_pass"):
        output = model(data_set)

# Print the profiling results
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
