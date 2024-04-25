import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src import utils
import wandb
from src import visualize
import pickle
from tqdm import tqdm

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

def image_gat_mp_trainer(model, 
                         embeddings, 
                         epochs, 
                         optimizer,
                         adjacency_tensor,
                         **kwargs):
    optim_params = kwargs.get("optim_params", {})
    optim = optimizer(model.parameters(), **optim_params)
    # with open(embedding_path, "rb") as f:
    #     embeddings = pickle.load(f)

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        optim.zero_grad()
        output, loss = model(embeddings, adjacency_tensor)

        loss.backward()
        optim.step()
        tqdm.write(f"Epoch: {epoch}, Loss: {loss.item()}")
    
    

