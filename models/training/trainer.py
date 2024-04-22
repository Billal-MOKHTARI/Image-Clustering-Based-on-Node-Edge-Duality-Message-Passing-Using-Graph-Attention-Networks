import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src import utils
import wandb

def train(model, 
          images, 
          epochs, 
          optimizer,
          primal_adjacency_tensor,
          dual_adjacency_tensor,
          dual_nodes,
          primal_weight = 1, 
          dual_weight = 1, 
          **kwargs):
    
    assert primal_weight >= 0 and primal_weight <= 1, "Primal weight must be between 0 and 1"
    assert dual_weight >= 0 and dual_weight <= 1, "Dual weight must be between 0 and 1"
    
    optimizer = optimizer(model.parameters(), **kwargs)
    for epoch in range(epochs):
        # Forward pass
        optimizer.zero_grad()
        result = model(images, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)

        primal_loss = utils.list_sum(result["primal"]["losses"])
        dual_loss = utils.list_sum(result["dual"]["losses"])
        conv_encoder_history = result["conv_encoder_history"]
        deconv_decoder_history = result["deconv_decoder_history"]
        
        loss = (primal_weight*primal_loss + dual_weight*dual_loss)/(primal_weight+dual_weight)

        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")