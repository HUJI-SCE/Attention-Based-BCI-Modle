import torch
import torch.nn as nn
import torch.autograd.profiler as profiler


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

        # decomposition of the metadata variables in topology
        if topology_metadata is None:
            topology_metadata = [(143, 77), (10, 5, 32)]

        channel_count, batch_width = topology_metadata[0]
        kernel_width, kernel_stride, fc_segment_length = topology_metadata[1]

        # Fully connected layer for each channel
        self.fc_layers = nn.ModuleList([nn.Linear(batch_width, fc_segment_length) for _ in range(channel_count)])

        # Custom Convolutional Segment
        self.conv_segment = CustomConvSegment(channel_count, batch_width, kernel_stride)

        # Concatenation layer
        self.concat_layer = nn.Linear(channel_count + (batch_width - kernel_width + 1), 128)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 1)  # 1 output node for binary classification

        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Fully connected layers for each channel
        fc_outputs = [fc(x[:, i, :]) for i, fc in enumerate(self.fc_layers)]

        # Convolutional segment
        conv_output = self.conv_segment(x)

        # Concatenate fully connected outputs and convolutional output
        concatenated = torch.cat(fc_outputs + [conv_output], dim=1)

        # Apply concatenation layer
        concatenated = nn.functional.relu(self.concat_layer(concatenated))

        # Apply fully connected layer with ReLU activation
        x = nn.functional.relu(self.fc1(concatenated))

        # Apply sigmoid activation for binary classification
        x = self.sigmoid(x)

        return x


class CustomConvSegment(nn.Module):
    def __init__(self, channel_count, batch_width, kernel_stride):
        super(CustomConvSegment, self).__init__()

        self.conv_layer = nn.Conv2d(1, 1, kernel_size=(channel_count, batch_width), stride=kernel_stride)

    def forward(self, x):
        # Expand dimensions for the single-channel convolution
        x = x.unsqueeze(1)

        # Apply the custom convolutional layer
        x = self.conv_layer(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        return x


# Set up input data
total_samples = 3_072_000
electrode_count = 128
kernel_count = 10
samples_per_batch = 150
input_data = torch.randn((total_samples - samples_per_batch, electrode_count, 1, samples_per_batch))

# Instantiate your model
model = MyModel(electrode_count, kernel_count, (1, samples_per_batch))

# Run the forward pass with profiling
with profiler.profile(record_shapes=True, use_cuda=False) as prof:
    with profiler.record_function("forward_pass"):
        output = model(input_data)

# Print the profiling results
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
