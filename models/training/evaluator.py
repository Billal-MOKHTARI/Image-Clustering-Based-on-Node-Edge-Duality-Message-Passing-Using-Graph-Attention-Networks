import sys
import os
from env import neptune_manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from networks.image_gat_message_passing import ImageGATMessagePassing
from data_loaders import data_loader
from typing import Union
import torch
import json
from src import utils
from models.networks.constants import DEVICE, FLOATING_POINT, ENCODING
import pickle
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def igmp_evaluator(embeddings: Union[torch.Tensor, str], 
                    row_index: Union[torch.List[str], str],
                    run: str,
                    checkpoint_path: str,
                    adjacency_tensor: Union[torch.Tensor, str] = None,
                    from_annotation_matrix: Union[None, str] = None,
                    weights_only: bool = False,
                    **kwargs):
    

  

    
    # Connect to Neptune
    run = neptune_manager.Run(run)

    # Load the embeddings.
    if isinstance(embeddings, str):
        embeddings = run.fetch_pkl_data(embeddings)
        
    # Load the row index
    if isinstance(row_index, str):
        row_index = run.fetch_pkl_data(row_index)

    # Load the adjacency tensor
    if isinstance(adjacency_tensor, str):
        adjacency_tensor = run.fetch_pkl_data(adjacency_tensor)
        
    if from_annotation_matrix is not None:
        adjacency_tensor, annot_row_index, annot_col_index = data_loader.annotation_matrix_to_adjacency_tensor(from_csv = from_annotation_matrix, transpose=True, sort="columns", index=row_index)
    assert row_index == annot_col_index, "Row indexes do not match"
    
    # Initialize the model
    graph_order = len(row_index)
    depth = adjacency_tensor.shape[0]
    model_args = kwargs.get("model_args", {})
    model = ImageGATMessagePassing(graph_order = graph_order, depth = depth, **model_args)
    

    
    # Load model checkpoint
    state_dict = run.load_model_checkpoint(checkpoint_path, map_location=DEVICE, encoding = ENCODING, weights_only = weights_only)
    model.load_state_dict(state_dict["model_state_dict"], strict=True)
    model.to(DEVICE)
    
    # Set the model to the evaluation mode

    _, hidden_losses, overall_loss = model(embeddings, adjacency_tensor)

    for i, hidden_loss in enumerate(hidden_losses):
        print(f"Hidden loss {i}: {hidden_loss}")

    print(f"Overall loss: {overall_loss}")
    
    # Remove the decoder and the loss from the model
    del model.decoder_layers
    del model.loss

    model.set_evaluation(True)

    data = model(embeddings, adjacency_tensor).detach().numpy()
    dataframe = pd.DataFrame(data, index=row_index)



    # pca = PCA(n_components=3)

    # # Fit the PCA model to your data
    # pca.fit(data)

    # # Transform the data to 3 principal components
    # data_pca = pca.transform(data)

    # # Transform the data to spherical coordinates
    # r = np.ones(data_pca.shape[0])  # Set radius to 1 for simplicity
    # theta = np.arccos(data_pca[:, 2] / r)  # Inclination angle
    # phi = np.arctan2(data_pca[:, 1], data_pca[:, 0])  # Azimuth angle

    # # Convert spherical coordinates to Cartesian coordinates
    # x = r * np.sin(theta) * np.cos(phi)
    # y = r * np.sin(theta) * np.sin(phi)
    # z = r * np.cos(theta)

    # # Plot the transformed data
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, c='r', marker='o')
    # ax.set_title('Spherical Representation of PCA')
    # plt.show()

    