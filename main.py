from models.networks.dual_gat_image_clustering import DualGATImageClustering
from models.networks import metrics, constants
from models.training import trainer
from src import visualize
from src import utils
from src import files_manager as fm
import pandas as pd
import torch
import numpy as np

if __name__ == "__main__":
    # training arguments
    epochs = 100
    num_images = 5
    image_dataset_path = "benchmark/datasets/test/shapes"
    primal_weight = 0.5
    dual_weight = 0.5

    # wandb arguments
    project = "Image Clustering Based on Node-Edge Duality Message Passing Using Graph Attention Networks"
    run_id_path = "configs/run_ids.bin"
    run_name = "DualGATImageClustering"


    # model arguments
    encoder_json_file_path = "configs/encoder.json"
    decoder_json_file_path = "configs/decoder.json"
    delimiter = "_"
    primal_criterion_weights=[1, 0.2, 0.3, 0.2, 1]
    dual_criterion_weights=[1, 0.2, 0.3, 0.2, 1]
    criterion_args = {"dim":1}

    # Load the images
    images = fm.read_images(image_dataset_path, num_images)
    image_size = images[0].shape

    # Load the adjacency tensor
    mat_square = pd.read_csv("benchmark/datasets/test/adjacency_matrix_square.csv", index_col=0, header=0, dtype=np.float32)
    mat_circle = pd.read_csv("benchmark/datasets/test/adjacency_matrix_circle.csv", index_col=0, header=0, dtype=np.float32)
    mat_triangle = pd.read_csv("benchmark/datasets/test/adjacency_matrix_triangle.csv", index_col=0, header=0, dtype=np.float32)

    primal_adjacency_tensor = torch.tensor(np.array([mat_square, mat_circle, mat_triangle]), dtype=constants.FLOATING_POINT)[:,:num_images,:num_images]
    n_objects = primal_adjacency_tensor.shape[0]
    num_images = primal_adjacency_tensor.shape[1]

    primal_index = utils.convert_list(mat_square.index.tolist(), np.float32, str)[:num_images]
    dual_index, dual_adjacency_tensor, dual_nodes = utils.create_dual_adjacency_tensor(primal_adjacency_tensor, 
                                                                                       primal_index, 
                                                                                       "_")

    # Load the encoder and decoder configurations
    encoder_args = fm.parse_encoder(encoder_json_file_path, network_type="encoder")
    decoder_args = fm.parse_encoder(decoder_json_file_path, network_type="decoder")

    primal_index = utils.convert_list(mat_square.index.tolist(), np.float32, str)[:num_images]
    dual_index, dual_adjacency_tensor, dual_nodes = utils.create_dual_adjacency_tensor(primal_adjacency_tensor, 
                                                                                       primal_index, 
                                                                                       delimiter)




    model = DualGATImageClustering(primal_index=primal_index, 
                                dual_index=dual_index, 
                                n_objects=n_objects, 
                                primal_criterion_weights=primal_criterion_weights, 
                                dual_criterion_weights=dual_criterion_weights,
                                image_size=image_size,
                                encoder_args=encoder_args, 
                                decoder_args=decoder_args, 
                                criterion=metrics.MeanCosineDistance,
                                criterion_args=criterion_args)

    # Connect to wandb
    visualize.connect_to_wandb(project=project,
                               run_id_path=run_id_path,
                               run_name=run_name)

    # Train the model
    trainer.train(model=model, 
                  images=images, 
                  epochs=epochs,
                  optimizer=torch.optim.Adam, 
                  primal_adjacency_tensor=primal_adjacency_tensor, 
                  dual_adjacency_tensor=dual_adjacency_tensor, 
                  dual_nodes=dual_nodes, 
                  primal_weight=primal_weight, 
                  dual_weight=dual_weight)
