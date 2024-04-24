from torch_model_manager import TorchModelManager, NeptuneManager

nm = NeptuneManager(project="Billal-MOKHTARI/Image-Clustering-based-on-Dual-Message-Passing",
                     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NGRlOTNiZC0zNGZlLTRjNWUtYWEyMC00NzEwOWJkOTRhODgifQ==",
                        run_ids_path="../../configs/run_ids.json")
run = nm.create_run("data_visualization", read_only=True)
print(run["embeddings"].download(destination="artifacts", progress_bar=True))