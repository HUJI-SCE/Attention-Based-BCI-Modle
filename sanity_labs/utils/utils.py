import os
import pickle
import torch


def store_pickle(obj, file_path_list):
    pickle_path = os.path.join(*file_path_list)
    with open(pickle_path, 'wb') as f:
        pickle.dump(obj, f)


def get_pickle(file_path_list):
    pickle_path = os.path.join(*file_path_list)
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


class DirMan:
    def __init__(self, dir_path, file_map, generator):
        self.dir_path = dir_path
        self.file_map = file_map
        self.generator = generator

    def __enter__(self):
        data = []
        self.absent_files = []

        # create directory if absent
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        # load existing files
        for source in self.file_map:
            current_gen = None
            for target in self.file_map[source]:
                target = os.path.join(self.dir_path, target)
                if os.path.exists(target):
                    current_gen = True
                    with open(target, 'rb') as f_data:
                        data += [torch.load(f_data)]

                # call generator to generate absent datasets
                else:
                    if current_gen is None:
                        data += [self.generator(source)]
                        self.absent_files += [(source, self.file_map[source], data[-1])]
                        break
                    else:
                        raise Exception(f'files are uncorrelated: dir = {self.dir_path}, absent files = {target}')
        return data

    def __exit__(self, exc_type, exc_value, traceback):
        # Save generated data into specified file paths
        for _, targets, datasets in self.absent_files:
            for target, dataset in zip(targets, datasets):
                target = os.path.join(self.dir_path, target)
                with open(target, 'wb') as f_data:
                    torch.save(dataset, f_data)


if __name__ == '__main__':
    pass

