from torch_model_manager import TorchModelManager, NeptuneManager
import os

nm = NeptuneManager(project="Billal-MOKHTARI/Image-Clustering-based-on-Dual-Message-Passing",
                     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NGRlOTNiZC0zNGZlLTRjNWUtYWEyMC00NzEwOWJkOTRhODgifQ==",
                        run_ids_path="../../configs/run_ids.json")
run = nm.create_run("data_visualization")
print(run["images/vgg16"].download(progress_bar=True))
