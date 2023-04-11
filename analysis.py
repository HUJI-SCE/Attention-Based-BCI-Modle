import torch


def calculate_similarity_tensor(data):
    torch_data = torch.tensor(data)

    similarity_matrices = torch.zeros(80, 128, 128)

    # loop over each sample in the data tensor
    for i in range(80):
        # extract the subset of 128 vectors for the current sample
        vectors = torch_data[i]

        # flatten the vectors tensor to size 128x256
        flat_vectors = vectors.view(128, -1)

        # calculate the dot product similarity matrix for the current sample
        similarity_matrix = torch.mm(flat_vectors, flat_vectors.t())

        # store the similarity matrix in the output tensor
        similarity_matrices[i] = similarity_matrix

    for i in range(80):
        min_val = torch.min(similarity_matrices[0])
        max_val = torch.max(similarity_matrices[0])
        similarity_matrices[0] = (similarity_matrices[0] - min_val) / (max_val - min_val)
    print(similarity_matrices[0])