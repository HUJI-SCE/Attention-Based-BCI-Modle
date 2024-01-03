import torch
import torch.nn as nn
import torch.autograd.profiler as profiler


# Define your model
class MyModel(nn.Module):
    def __init__(self, _electrode_count, _kernel_count, _samples_per_batch):
        super(MyModel, self).__init__()
        self.conv_layer = nn.Conv2d(_electrode_count, _kernel_count, kernel_size=_samples_per_batch)

    def forward(self, x):
        return self.conv_layer(x)


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
