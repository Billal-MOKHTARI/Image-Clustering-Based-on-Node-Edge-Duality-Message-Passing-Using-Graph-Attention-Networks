import wandb
import os
import matplotlib.pyplot as plt
from typing import Union
from . import files_manager as fm
import plotly.graph_objs as go

def create_directory_tree(directory_structure: str, parent_path: str):
    """
    Recursively creates folders based on the provided directory structure dictionary.

    Parameters:
    - directory_structure (dict): The dictionary representing the directory structure.
    - parent_path (str): The parent path where the folders should be created. Default is the current directory.
    """
    for folder_name, subfolders in directory_structure.items():
        folder_path = os.path.join(parent_path, folder_name)

        os.makedirs(folder_path, mode=777, exist_ok=True)

        if subfolders:
            create_directory_tree(subfolders, folder_path)

def connect_to_wandb(project: str, 
                    run_id_path: str,
                    run_name: str) -> None:
    
    # Additional runtime checks if needed
    assert isinstance(project, str), "project_name should be a string"
    assert isinstance(run_id_path, Union[str, None]), "run_id_path should be a string"
    assert isinstance(run_name, Union[str, None]), "run_name should be a string"

    run_id = None
    resume = None 
    
    try:
        run_ids = fm.load_data_from_path(run_id_path)
        
        if run_ids is None:
            run_ids = dict()

        if run_name in run_ids.keys():
            run_id = run_ids[run_name]
            resume = "must"
    except:
        print(f"{run_id_path} has been created")

    finally:
    
        wandb.init(project = project, name = run_name, id = run_id, resume = resume)

        # If the run is created for the first time, we will associate the run id to the run name
        # We want that this file could not be directly accessed by the user
        not_exist_run_name = (os.path.exists(run_id_path)) \
                        and (run_name not in run_ids.keys()) \
                        or (not os.path.exists(run_id_path))
        
        if not_exist_run_name:
            run_ids[run_name] = wandb.run.id
            fm.dump_data(run_ids, run_id_path)

def show_images(images, num_rows, num_cols, titles=None, scale=1.5, grayscale=False):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if grayscale and img.size(0) == 1:
            print(img.shape)
            print(img.numpy().reshape(1, img.shape[0], img.shape[1]))
            ax.imshow(img.numpy().reshape(1, img.shape[0], img.shape[1]), cmap='gray')
        else:
            ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def plot_clusters(data, cluster_column='cluster'):
    """
    Plot clusters in 3D using Plotly.

    Parameters:
    - data (pd.DataFrame): Data with cluster labels.

    Returns:
    - None
    """
    fig = go.Figure()

    for cluster_label in data[cluster_column].unique():
        cluster_data = data[data[cluster_column] == cluster_label]
        fig.add_trace(go.Scatter3d(
            x=cluster_data['x'],
            y=cluster_data['y'],
            z=cluster_data['z'],
            mode='markers',
            marker=dict(
                size=5,
                opacity=0.8,
            ),
            name=f'Cluster {cluster_label}'
        ))

    fig.update_layout(
        title='Spherical Representation of PCA with Clusters',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        )
    )

    fig.show()