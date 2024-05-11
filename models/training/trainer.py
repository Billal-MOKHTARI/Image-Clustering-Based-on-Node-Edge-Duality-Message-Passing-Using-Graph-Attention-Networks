import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src import utils
import wandb
from src import visualize
import pickle
from tqdm import tqdm
from env import neptune_manager
from torch import nn
import torch
from typing import Union, List, Dict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from networks import image_gat_message_passing as igmp
from data_loaders import data_loader
from time import time
import numpy as np
import asyncio
from models.networks.constants import DEVICE, FLOATING_POINT, ENCODING
import json
from models.networks import metrics
from torch_model_manager import TorchModelManager

def train(model, 
          images, 
          epochs, 
          optimizer,
          primal_adjacency_tensor,
          dual_adjacency_tensor,
          dual_nodes,
          primal_weight = 1, 
          dual_weight = 1, 
          use_wandb = True,
          **kwargs):
    
    assert primal_weight >= 0 and primal_weight <= 1, "Primal weight must be between 0 and 1"
    assert dual_weight >= 0 and dual_weight <= 1, "Dual weight must be between 0 and 1"
    wandb_args = kwargs.get("wandb_args", {})

    optimizer = optimizer(model.parameters(), **kwargs)
    for epoch in range(epochs):
        # Forward pass
        optimizer.zero_grad()
        result = model(images, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)

        primal_loss = utils.list_sum(result["primal"]["losses"])
        dual_loss = utils.list_sum(result["dual"]["losses"])


        loss_log = {}
        for i, res in enumerate(result["primal"]["losses"]):
            loss_log[f"primal loss (layer {i})"] = res

        for i, res in enumerate(result["dual"]["losses"]):
            loss_log[f"dual loss (layer {i})"] = res

        loss_log.update({"primal loss": primal_loss, "dual loss": dual_loss})



        conv_encoder_history = result["conv_encoder_history"]
        deconv_decoder_history = result["deconv_decoder_history"]
        
        loss = (primal_weight*primal_loss + dual_weight*dual_loss)/(primal_weight+dual_weight)

        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

        loss_log["loss"] = loss.item()

        # Log the loss
        if use_wandb:
                        # Connect to wandb
            visualize.connect_to_wandb(**wandb_args)
            wandb.log(loss_log, step = epoch, commit = True)

def image_gat_mp_trainer(embeddings: Union[torch.Tensor, Dict], 
                         row_index: Union[torch.List[str], str, Dict],
                         epochs: int, 
                         optimizer: torch.optim.Optimizer,
                         run: str,
                         adjacency_tensor: Union[torch.Tensor, str, Dict] = None,
                         from_annotation_matrix: Union[None, str, Dict] = None,
                         weights_only: bool = False,
                         **kwargs):
    
    torch.use_deterministic_algorithms(True)
    
    run_args = kwargs.get("run_args", {})

    # Connect to Neptune
    run = neptune_manager.Run(run, **run_args)

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
        
    # Load the annotation matrix if it is provided
    if from_annotation_matrix is not None:
        # Load the annotation matrix from Neptune
        if isinstance(from_annotation_matrix, Dict):
            from_annotation_matrix = from_annotation_matrix["path"]
            from_run_metadata = from_annotation_matrix["from_run_metadata"]
            
            # The annotation matrix is stored in the run metadata
            if from_run_metadata:
                from_annotation_matrix = run.fetch_pkl_data(from_annotation_matrix)
            
            # The annotation matrix is stored in Neptune project metadata
            else:
                from_annotation_matrix = neptune_manager.fetch_pkl_data(from_annotation_matrix)
                
            adjacency_tensor, annot_row_index, annot_col_index = data_loader.annotation_matrix_to_adjacency_tensor(matrix = from_annotation_matrix, transpose=True, sort="columns", index=row_index)
        
        # The annotation matrix is stored in a local file
        if isinstance(from_annotation_matrix, str):
            adjacency_tensor, annot_row_index, annot_col_index = data_loader.annotation_matrix_to_adjacency_tensor(from_csv = from_annotation_matrix, transpose=True, sort="columns", index=row_index)

    # Check if the row indexes match
    assert row_index == annot_col_index, "Row indexes do not match"
  
    # Get the graph order and depth
    graph_order = len(row_index)
    depth = adjacency_tensor.shape[0]

    # Get the model arguments
    model_args = kwargs.get("model_args", {})
    namespace = kwargs.get("namespace", "training")
    checkpoint_namespace = kwargs.get("checkpoint_namespace", "checkpoints")
    hyperparam_namespace = kwargs.get("hyperparameter_namespace", "hyperparameters")
    loss_namespace = kwargs.get("loss_namespace", "losses")
    initial_parameter_namespace = kwargs.get("initial_parameter_namespace")
    keep = kwargs.get("keep", 3)

    log_freq = kwargs.get("log_freq", 100)


    # Define the model
    model = igmp.ImageGATMessagePassing(graph_order = graph_order, depth = depth, **model_args)
    tmm = TorchModelManager(model)

    try:
        weights = run.fetch_pkl_data(initial_parameter_namespace)
        model.load_state_dict(weights)
    except:
        run.log_files(data=model.state_dict(), namespace=initial_parameter_namespace, extension=".pkl", wait=True)

  
    # Initialize the optimizer
    optim_params = kwargs.get("optim_params", {})
    optim = optimizer(model.parameters(), **optim_params)
    
    # Define initial values
    current_epoch = -1
    min_loss = np.inf
    # Load the model checkpoint
    try:
        path = utils.get_max_element(run.fetch_files(os.path.join(namespace, checkpoint_namespace)), delimiter="_")
        full_path = os.path.join(namespace, checkpoint_namespace, path)
        state_dict = run.load_model_checkpoint(full_path, map_location=DEVICE, encoding = ENCODING, weights_only = weights_only)
        print("Loaded checkpoint : ", full_path)

        if state_dict is None:
            run.delete_data(full_path, wait=False)
            path = utils.get_max_element(run.fetch_files(os.path.join(namespace, checkpoint_namespace)), delimiter="_")
            full_path = os.path.join(namespace, checkpoint_namespace, path)

            state_dict = run.load_model_checkpoint(full_path, map_location=DEVICE, encoding = ENCODING, weights_only = weights_only)
            
        current_epoch = state_dict["epoch"]
        min_loss = state_dict["loss"]

        model.to(DEVICE)
        model.load_state_dict(state_dict["model_state_dict"])
        
        optim = optimizer(model.parameters(), **optim_params)
        optim.load_state_dict(state_dict["optimizer_state_dict"])


    except:
        pass
    

    

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, 
                                                  last_epoch=current_epoch, 
                                                  lr_lambda=lambda epoch: optim_params['lr'])
    
    path_metric = os.path.join(namespace, loss_namespace, f"{model_args['loss']().__class__.__name__}")
    path_hidden_layers = []
    for i in range(len(model_args["layer_sizes"])):
        path_hidden_layers.append(os.path.join(namespace, loss_namespace, f"Hidden Layer {model_args['loss']().__class__.__name__} (size = {model_args['layer_sizes'][i]})"))


    # Log hyperparameters
    hyperparameters = {"model": model.__class__.__name__,
                       "model_args": model_args,
                       "optimizer": optim.__class__.__name__, 
                       "optim_params": optim_params,
                       "log_freq": log_freq,
                       "graph": {"order": graph_order, "node_dimension": depth},
                       }
    run.log_hyperparameters(hyperparams = hyperparameters,
                            namespace = os.path.join(namespace, hyperparam_namespace))

    for epoch in tqdm(range(current_epoch+1, current_epoch+epochs+1), desc="Training", unit="epoch"):

        optim.zero_grad()
        output, hidden_losses, overall_loss = model(embeddings, adjacency_tensor)
        
        overall_loss.backward()
        optim.step()
        scheduler.step()


       
        # Log the losses
        for loss, path_hl in zip(hidden_losses, path_hidden_layers):
            run.track_metric(namespace = path_hl,
                             metric = loss.item(),
                             step = None,
                             timestamp = time(),
                             wait=False)
        
        run.track_metric(namespace = path_metric, 
                         metric = overall_loss.item(),
                         step=None,
                         timestamp = time(),
                         wait=False)
        
        if overall_loss.item() < min_loss:
            min_loss = overall_loss.item()
    
            if epoch % log_freq == 0 and epoch !=0 :
                checkpoint_path = os.path.join(namespace, checkpoint_namespace)
              
                run.log_checkpoint(model = model, 
                                optimizer=optim,
                                loss = overall_loss.item(),
                                epoch = epoch,
                                namespace = os.path.join(checkpoint_path, f"chkpt_epoch_{epoch}"),
                                wait = True,
                                keep = keep)

    run.stop_run()

