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
from typing import Union, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from networks import image_gat_message_passing as igmp
from data_loaders import data_loader
from time import time
 
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

def image_gat_mp_trainer(embeddings: Union[torch.Tensor, str], 
                         row_index: Union[torch.List[str], str],
                         epochs: int, 
                         optimizer: torch.optim.Optimizer,
                         run: str,
                         adjacency_tensor: Union[torch.Tensor, str] = None,
                         from_annotation_matrix: Union[None, str] = None,
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
  
    graph_order = len(row_index)
    depth = adjacency_tensor.shape[0]

    # Get the model arguments
    model_args = kwargs.get("model_args", {})
    namespace = kwargs.get("namespace", "training")
    checkpoint_namespace = kwargs.get("checkpoint_namespace", "checkpoints")
    loss_namespace = kwargs.get("loss_namespace", "losses")
    
    log_freq = kwargs.get("log_freq", 100)


    # Define the model
    model = igmp.ImageGATMessagePassing(graph_order = graph_order, depth = depth, **model_args)

    # Initialize the optimizer
    optim_params = kwargs.get("optim_params", {})
    optim = optimizer(model.parameters(), **optim_params)

    current_epoch = -1
    
    try:
        path = utils.get_max_element(run.fetch_files(os.path.join(namespace, checkpoint_namespace)), delimiter="_")
        state_dict = run.load_model_checkpoint(os.path.join(namespace, checkpoint_namespace, path))
        
        model.load_state_dict(state_dict["model_state_dict"])
        optim.load_state_dict(state_dict["optimizer_state_dict"])
        current_epoch = state_dict["epoch"]
    except:
        pass
    
    path_metric = os.path.join(namespace, loss_namespace, f"{model_args['loss']().__class__.__name__}")
    path_hidden_layers = []
    for i in range(len(model_args["layer_sizes"])):
        path_hidden_layers.append(os.path.join(namespace, loss_namespace, f"Hidden Layer {model_args['loss']().__class__.__name__} (size = {model_args['layer_sizes'][i]})"))

    for epoch in tqdm(range(current_epoch+1, current_epoch+epochs+1), desc="Training", unit="epoch"):

        optim.zero_grad()
        output, hidden_losses, overall_loss = model(embeddings, adjacency_tensor)
        
        overall_loss.backward()
        optim.step()
        
        # Log the losses
        for loss, path_hl in zip(hidden_losses, path_hidden_layers):
            run.track_metric(namespace = path_hl,
                             metric = loss.item(),
                             step=epoch,
                             timestamp = time(),
                             wait=False)
        
        run.track_metric(namespace = path_metric, 
                         metric = overall_loss.item(),
                         step=epoch,
                         timestamp = time(),
                         wait=False)
        
        if epoch % log_freq == 0:
            run.log_checkpoint(model = model, 
                               optimizer=optim,
                               loss = model_args["loss"],
                               epoch = epoch,
                               namespace = os.path.join(namespace, checkpoint_namespace,f"chkpt_epoch_{epoch}"))

