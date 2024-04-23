from torch import nn
from .dual_message_passing import DualMessagePassing
from .custom_layers import Encoder2D, Decoder2D


class DualGATImageClustering(nn.Module):
    def __init__(self, 
                primal_index,
                dual_index,
                n_objects,
                criterion = nn.MSELoss,
                primal_criterion_weights = [1, 1, 1, 1, 1],
                dual_criterion_weights = [1, 1, 1, 1, 1],
                primal_mp_layer_inputs=[64, 32, 16, 8, 4],  
                dual_mp_layer_inputs=[64, 32, 16, 8],
                delimiter="_",
                with_replacement=True,
                **kwargs):
        """
        DualGATImageClustering model for image clustering based on node-edge duality message passing using Graph Attention Networks.
        
        Args:
            primal_index (list): List of primal node indices.
            dual_index (list): List of dual node indices.
            n_objects (int): Number of objects in the graph.
            backbone (str): Backbone model architecture for image encoding. Default is 'vgg16'.
            in_image_size (tuple): Input image size. Default is (3, 224, 224).
            margin_expansion_factor (int): Margin expansion factor for image encoding. Default is 6.
            primal_mp_layer_inputs (list): List of input sizes for primal message passing layers. Default is [728, 600, 512, 400, 256].
            dual_mp_layer_inputs (list): List of input sizes for dual message passing layers. Default is [1000, 728, 600, 512, 400].
            delimiter (str): Delimiter used for creating dual node indices. Default is "_".
            **kwargs: Additional keyword arguments for image encoder and dual message passing layers.
        """
        super(DualGATImageClustering, self).__init__()
        
        # Extract additional keyword arguments
        dual_message_passing_args = kwargs.get("dual_message_passing_args", {})
        criterion_args = kwargs.get("criterion_args", {})
        self.encoder_args = kwargs.get("encoder_args", {})
        self.decoder_args = kwargs.get("decoder_args", {})
        self.with_replacement = with_replacement
        
        # Store input size and primal index
        self.primal_index = primal_index 
        self.dual_index = dual_index
        self.delimiter = delimiter 
        self.criterion = criterion(**criterion_args)
        self.primal_criterion_weights = primal_criterion_weights
        self.dual_criterion_weights = dual_criterion_weights

        # Image encoder parameters

        self.enc_primal_mp_layer_inputs = primal_mp_layer_inputs
        self.enc_dual_mp_layer_inputs = [n_objects] + dual_mp_layer_inputs
        self.dec_primal_mp_layer_inputs = self.enc_primal_mp_layer_inputs[::-1]
        self.dec_dual_mp_layer_inputs = self.enc_dual_mp_layer_inputs[::-1]

        if self.with_replacement:
            
            self.primal_depths = [n_objects] + dual_mp_layer_inputs
            self.dual_depths = [n_objects] + primal_mp_layer_inputs[1:]
        else:
            self.primal_depths = [n_objects]*(len(dual_mp_layer_inputs)+1)
            self.dual_depths = [n_objects]*(len(dual_mp_layer_inputs)+1)

        # Create encoder and decoder message passing layers
        self.enc_dmp_layers = []
        self.dec_dmp_layers = []

        self.image_encoder = self.get_image_encoder()
        self.image_decoder = self.get_image_decoder()

        for i in range(len(self.enc_primal_mp_layer_inputs)-1):
            self.enc_dmp_layers.append(DualMessagePassing(primal_in_features=self.enc_primal_mp_layer_inputs[i], 
                                                      primal_out_features=self.enc_primal_mp_layer_inputs[i+1], 
                                                      primal_index=self.primal_index,
                                                      primal_depth=self.primal_depths[i],
                                                    
                                                      dual_in_features=self.enc_dual_mp_layer_inputs[i],
                                                      dual_out_features=self.enc_dual_mp_layer_inputs[i+1],
                                                      dual_index=self.dual_index,
                                                      dual_depth=self.dual_depths[i],
                                                      layer_index=f"encoder_{i}",
                                                      delimiter=self.delimiter,
                                                      with_replacement=self.with_replacement,
                                                      **dual_message_passing_args))
            
        for i in range(len(self.dec_primal_mp_layer_inputs)-1):
            self.dec_dmp_layers.append(DualMessagePassing(primal_in_features=self.dec_primal_mp_layer_inputs[i], 
                                                      primal_out_features=self.dec_primal_mp_layer_inputs[i+1], 
                                                      primal_index=self.primal_index,
                                                      primal_depth=self.dec_dual_mp_layer_inputs[i] if self.with_replacement else self.primal_depths[i],
                                                    
                                                      dual_in_features=self.dec_dual_mp_layer_inputs[i],
                                                      dual_out_features=self.dec_dual_mp_layer_inputs[i+1],
                                                      dual_index=self.dual_index,
                                                      dual_depth=self.dec_primal_mp_layer_inputs[i] if self.with_replacement else self.dual_depths[i],
                                                      layer_index=f"decoder_{i}",
                                                      delimiter=self.delimiter,
                                                      with_replacement=self.with_replacement,
                                                      **dual_message_passing_args))

    def get_image_encoder(self):
        return Encoder2D(**self.encoder_args)


    def get_image_decoder(self):
        return Decoder2D(**self.decoder_args)


    def encoder(self, primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes):
        encoder_history = []
 
        for layer in self.enc_dmp_layers:
            tmp = {"primal_nodes": primal_nodes, "dual_nodes": dual_nodes}
            encoder_history.append(tmp)

            result = layer(primal_nodes, dual_nodes, primal_adjacency_tensor, dual_adjacency_tensor)
            
            primal_nodes, primal_adjacency_tensor = result["primal"]["nodes"], result["primal"]["adjacency_tensor"]
            dual_nodes, dual_adjacency_tensor = result["dual"]["nodes"], result["dual"]["adjacency_tensor"]
            
            
        tmp = {"primal_nodes": primal_nodes, "dual_nodes": dual_nodes}
        encoder_history.append(tmp)
        
        return primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor, encoder_history

    def decoder(self, primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes):

        decoder_history = []

        for layer in self.dec_dmp_layers:

            tmp = {"primal_nodes": primal_nodes, "dual_nodes": dual_nodes}
            decoder_history.append(tmp)

            result = layer(primal_nodes, dual_nodes, primal_adjacency_tensor, dual_adjacency_tensor)

            primal_nodes, primal_adjacency_tensor = result["primal"]["nodes"], result["primal"]["adjacency_tensor"]
            dual_nodes, dual_adjacency_tensor = result["dual"]["nodes"], result["dual"]["adjacency_tensor"]

        tmp = {"primal_nodes": primal_nodes, "dual_nodes": dual_nodes}
        decoder_history.append(tmp)


        return primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor, decoder_history
        
    def forward(self, imgs, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes):
        """
        Forward pass of the DualGATImageClustering model.
        
        Args:
            imgs (torch.Tensor): Input images.
            primal_adjacency_tensor (torch.Tensor): Primal adjacency tensor.
            dual_adjacency_tensor (torch.Tensor): Dual adjacency tensor.
            dual_nodes (torch.Tensor): Dual nodes.
        
        Returns:
            tuple: Tuple containing primal nodes, primal adjacency tensor, dual nodes, and dual adjacency tensor.
        """
        # Encode images to embeddings
        
        encoder_history = []

        primal_nodes, indices, conv_encoder_history = self.image_encoder(imgs)
        primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor, encoder_history = self.encoder(primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)
        primal_nodes, primal_adjacency_tensor, dual_nodes, dual_adjacency_tensor, decoder_history = self.decoder(primal_nodes, primal_adjacency_tensor, dual_adjacency_tensor, dual_nodes)
 
        primal_nodes, deconv_decoder_history = self.image_decoder(primal_nodes, indices[::-1])
        
        decoder_history = decoder_history[::-1]

        encoder_history.pop()
        decoder_history.pop()

        # Compute primal reconstruction loss
        primal_losses = []
        dual_losses = []

        num_layers = len(encoder_history)
        for i in range(num_layers):
            enc_primal_node = encoder_history[i]["primal_nodes"]
            dec_primal_node = decoder_history[i]["primal_nodes"]
            primal_losses.append(self.primal_criterion_weights[i] * self.criterion(enc_primal_node, dec_primal_node))

            enc_dual_node = encoder_history[i]["dual_nodes"]
            dec_dual_node = decoder_history[i]["dual_nodes"]
            dual_losses.append(self.dual_criterion_weights[i] * self.criterion(enc_dual_node, dec_dual_node))

        result = {
            "primal": {
                "nodes": primal_nodes,
                "adjacency_tensor": primal_adjacency_tensor,
                "losses": primal_losses},
            "dual": {
                "nodes": dual_nodes,
                "adjacency_tensor": dual_adjacency_tensor,
                "losses": dual_losses
            },
            "conv_encoder_history": conv_encoder_history,
            "deconv_decoder_history": deconv_decoder_history[::-1]
        }

        return result
