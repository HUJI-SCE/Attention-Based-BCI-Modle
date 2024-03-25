import os
import pickle

import torch.autograd.profiler as profiler


def store_pickle(obj, file_path_list):
    pickle_path = os.path.join(*file_path_list)
    with open(pickle_path, 'wb') as f:
        pickle.dump(obj, f)


def get_pickle(file_path_list):
    pickle_path = os.path.join(*file_path_list)
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def timestamp():
    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    # dd_mm_YY-H:M:S
    dt_string = now.strftime("%d_%m_%Y-%H:%M:%S")
    return dt_string


def profile_model(model, data_set):
    print(model)
    # Run the forward pass with profiling
    with profiler.profile(record_shapes=True, use_cuda=False) as prof:
        with profiler.record_function("forward_pass"):
            output = model(data_set)

    # Print the profiling results
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))


if __name__ == '__main__':
    pass
