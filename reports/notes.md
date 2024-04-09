* We cannot apply ISOMAP or any manifold learning meathod to reduce the dimensionaly of the adjacency matrix since the order of nodes is crucial and in manifold learning the set of points describe a specific object. Conversely, our matrix is very well defined, if we add any point to the tensor, it becomes incoherent.
* Reducing the features from high dimensional space to $1$, costs **highly loss of information**.
* **Solution** : reduce partially the tensor to $d<\log_{2}(n)$ where $n$ is the depth of the tensor. Then carry on the training with the reduced tensor.
