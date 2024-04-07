from torch import nn

class IntermediateEncoder(nn.Module):
    def __init__(self, out_features, **kwargs):
        super(IntermediateEncoder, self).__init__()
        self.out_features = out_features
        
        