import pandas as pd
import torch
import plotly.express as px



def calculate_similarity(vec_1, vec_2):
    return torch.mm(vec_1, vec_2)


def calculate_similarity_tensor(data):
    torch_data = torch.tensor(data)

    similarity_matrices = torch.zeros(80, 128, 128)

    # loop over each sample in the data tensor
    for i in range(80):
        vectors = torch_data[i]
        flat_vectors = vectors.view(128, -1)
        similarity_matrix = torch.mm(flat_vectors, flat_vectors.t())
        similarity_matrices[i] = similarity_matrix

    for i in range(80):
        min_val = torch.min(similarity_matrices[i])
        max_val = torch.max(similarity_matrices[i])
        similarity_matrices[i] = (similarity_matrices[i] - min_val) / (max_val - min_val)
    return similarity_matrices


def visualize_similarity(index):
    heatmap = calculate_similarity_tensor(pd.read_pickle('input/clip_eeg.pickle'))[index]

    fig = px.imshow(heatmap, color_continuous_scale='viridis')

    # get mean and variance values
    mean_value = heatmap.mean().item()
    variance_value = heatmap.var().item()

    fig.update_layout(
        coloraxis_colorbar=dict(title='Values'),
        annotations=[
            dict(
                x=0.5,
                y=-0.1,
                xref='paper',
                yref='paper',
                text="Mean Value: {:.2f}".format(mean_value),
                showarrow=False,
                font=dict(size=14),
            ),
            dict(
                x=0.5,
                y=-0.15,
                xref='paper',
                yref='paper',
                text="Variance: {:.2f}".format(variance_value),
                showarrow=False,
                font=dict(size=14),
            ),
        ],
    )

    return fig
