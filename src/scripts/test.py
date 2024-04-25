from torch_model_manager import TorchModelManager, NeptuneManager
import os
import pandas as pd
import numpy as np
from neptune.types import File
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath)

from torchvision import models, transforms
import pipelines
from project_consts import NEPTUNE_MANAGER, DATA_VISUALIZATION_RUN



model = models.vgg19(pretrained=True)
del model.classifier[-1]
del model.classifier[-1]
del model.classifier[-1]
   
data_path = "../../benchmark/datasets/agadez"
neptune_workspace = "embeddings/agadez_vgg19_embeddings"

# pipelines.create_embeddings(model, NEPTUNE_MANAGER, run, neptune_workspace, data_path, torch_transforms=[transforms.Resize((512, 512))])
# nm.delete_data(run, ['embeddings'])