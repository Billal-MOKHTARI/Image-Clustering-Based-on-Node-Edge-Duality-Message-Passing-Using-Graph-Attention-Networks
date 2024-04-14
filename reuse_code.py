import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# model = DualMessagePassing(
#                             primal_in_features=3, 
#                             primal_out_features=7, 
#                             depth=3,
#                             primal_index=["n1", "n2", "n3", "n4"],
#                             dual_in_features=5,
#                             dual_out_features=3,
    
#                             dual_index=["n1_n2", "n1_n3", "n2_n4"], 
#                             layer_index=1,
#                             delimiter="_",
#                             node_message_passing_args={"batch_norm": {"momentum": 0.1}, "activation": {"layer": nn.ReLU, "args": {}}},
#                             edge_message_passing_args={"batch_norm": {"momentum": 0.1}, "activation": {"layer": nn.ReLU, "args": {}}}
#                             )

# node_x = torch.tensor(np.array([[1, 2, 3], [4, 6, 5], [7, 8, 9], [10, 11, 12]]), dtype=torch.float32)
# edge_x = torch.tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]), dtype=torch.float32)
# primal_adjacency_tensor = torch.tensor(np.array([[[1, 1, 1, 0], 
#                                                   [1, 1, 0, 0], 
#                                                   [1, 0, 1, 0], 
#                                                   [0, 0, 0, 1]], 
#                                                  [[1, 1, 1, 0], 
#                                                   [1, 1, 0, 1], 
#                                                   [1, 0, 1, 0], 
#                                                   [0, 1, 0, 1]], 
#                                                  [[1, 1, 0, 0], 
#                                                   [1, 1, 0, 1], 
#                                                   [0, 0, 1, 0], 
#                                                   [0, 1, 0, 1]]]), dtype=torch.float32)
# dual_adjacency_tensor = torch.tensor(np.array([[[1, 2, 3], 
#                                                 [2, 1, 1], 
#                                                 [3, 1, 0]], 
#                                                [[1, 0, 1], 
#                                                 [0, 1, 0], 
#                                                 [1, 0, 1]], 
#                                                [[1, 0, 1], 
#                                                 [0, 1, 1], 
#                                                 [1, 1, 1]]]), dtype=torch.float32)
# result = model(node_x, edge_x, primal_adjacency_tensor, dual_adjacency_tensor)
# print(result)