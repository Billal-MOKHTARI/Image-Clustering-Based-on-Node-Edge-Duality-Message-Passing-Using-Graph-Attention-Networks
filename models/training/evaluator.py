import sys
import os
from env import neptune_manager
from . import clustering

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src import visualize
from networks.image_gat_message_passing import ImageGATMessagePassing
from data_loaders import data_loader
from typing import Union, Dict
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
                    clustering_method = None,
                    **kwargs):
    
    # Connect to Neptune
    run = neptune_manager.Run(run)

  # Load the embeddings.
    if isinstance(embeddings, str):
        with open(embeddings, "rb") as f:
            embeddings = pickle.load(f)
    elif isinstance(embeddings, dict):
        embedding_path = embeddings["path"]
        from_run_metadata = embeddings["from_run_metadata"]

        if from_run_metadata :
            embeddings = run.fetch_pkl_data(embedding_path)
        else:
            embeddings = neptune_manager.fetch_pkl_data(embedding_path)

    # Load the row index
    if isinstance(row_index, str):
        with open(row_index, "rb") as f:
            row_index = pickle.load(f)
    elif isinstance(row_index, dict):
        row_index_path = row_index["path"]
        from_run_metadata = row_index["from_run_metadata"]

        if from_run_metadata:
            row_index = run.fetch_pkl_data(row_index_path)
        else:
            row_index = neptune_manager.fetch_pkl_data(row_index_path)

    # Load the adjacency tensor
    if isinstance(adjacency_tensor, str):
        with open(adjacency_tensor, "rb") as f:
            adjacency_tensor = pickle.load(f)
    elif isinstance(adjacency_tensor, dict):
        adjacency_tensor_path = adjacency_tensor["path"]
        from_run_metadata = adjacency_tensor["from_run_metadata"]

        if from_run_metadata:
            adjacency_tensor = run.fetch_pkl_data(adjacency_tensor_path)
        else:
            adjacency_tensor = neptune_manager.fetch_pkl_data(adjacency_tensor_path)

    if from_annotation_matrix is not None:
        adjacency_tensor, annot_row_index, annot_col_index = data_loader.annotation_matrix_to_adjacency_tensor(from_csv = from_annotation_matrix, transpose=True, sort="columns", index=row_index)
    assert row_index == annot_col_index, "Row indexes do not match"
    
    # Initialize the model
    graph_order = len(row_index)
    depth = adjacency_tensor.shape[0]
    model_args = kwargs.get("model_args", {})
    model = ImageGATMessagePassing(graph_order = graph_order, depth = depth, **model_args)
    output_args = kwargs.get("output_args", {})
    clustering_args = kwargs.get("clustering_args", {})
    
    # Load model checkpoint
    state_dict = run.load_model_checkpoint(checkpoint_path, map_location=DEVICE, encoding = ENCODING, weights_only = weights_only)
    model.load_state_dict(state_dict["model_state_dict"], strict=True)
    model.to(DEVICE)
    
    # Set the model to the evaluation mode
    with torch.no_grad():
        _, hidden_losses, overall_loss = model(embeddings, adjacency_tensor)
    
    # Remove the decoder and the loss from the model
    del model.decoder_layers
    del model.loss

    model.set_evaluation(True)
    with torch.no_grad():
        data = model(embeddings, adjacency_tensor).detach().numpy()
    dataframe = pd.DataFrame(data, index=row_index)
    dataframe = clustering.clustering(method = clustering_method, data = dataframe, **clustering_args)
    clusters = dataframe["cluster"].values
    print(f"{len(clusters[clusters == -1])/len(clusters)*100:.4f} %")
    
    # Save the outputs in neptune
    # if output_args is not None:
    #     log_dataframe_args = output_args.get("log_dataframe", {})
        
    #     run.log_dataframe(dataframe = dataframe, 
    #                       df_format = True,
    #                       csv_format = True,
    #                       **log_dataframe_args)
    #     for hidden_loss, layer_size in zip(hidden_losses, model_args["layer_sizes"]):
    #         run.log_args(namespace=os.path.join(output_args["metrics"], f"hidden_loss (layer_size = {layer_size})"), args=hidden_loss)
            

    pca = PCA(n_components=3)

    # Fit the PCA model to your data
    pca.fit(data)

    # Transform the data to 3 principal components
    data_pca = pca.transform(data)

    data_viz = pd.DataFrame({'x': data_pca[:, 0], 'y': data_pca[:, 1], 'z': data_pca[:, 2], 'cluster': dataframe['cluster']})

    visualize.plot_clusters(data_viz, cluster_column='cluster')
    

    