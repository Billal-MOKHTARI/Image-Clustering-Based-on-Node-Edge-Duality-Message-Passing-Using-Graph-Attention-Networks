
import streamlit as st
from tempfile import NamedTemporaryFile
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.training import clustering
from models.training.evaluator import igmp_evaluator
from models.networks import metrics
import json 
from torch import nn
from src import visualize
from sklearn.decomposition import PCA
import pandas as pd


st.set_page_config(layout="wide")

def evaluation_form(config=None):

    submitted = False
    st.subheader('Use custom loss')
    use_custom_loss = st.checkbox('Use custom loss', key='use_custom_loss_checkbox')

    st.subheader('Model parameters')

    layer_sizes = st.multiselect('Layer sizes', st.session_state.get('layer_sizes', [512, 384, 256, 128]),  default=[512, 384, 256, 128], key='layer_sizes_multiselect')
    new_layer_size = st.text_input('Add new layer size', key='new_layer_size_text_input')
    if st.button('Add', key='add_new_layer_size_button'):
        if new_layer_size.strip() != '':
            layer_sizes.append(int(new_layer_size))
            st.session_state.layer_sizes = list(set(layer_sizes))
            st.experimental_rerun()

    loss = st.selectbox('Loss', ['MeanCosineDistance', 'MSELoss'], index=0, key='loss_selectbox')
    loss_coeffs = st.multiselect('Loss coefficients', st.session_state.get('loss_coeffs', [1.0, 0.8, 0.6]), default=[1.0, 0.8, 0.6], key='loss_coefficients_multiselect')
    new_loss_coeff = st.text_input('Add new loss coeff', key='loss_coefficients_text_input')
    if st.button('Add', key="loss_coefficients_button"):
        if new_loss_coeff.strip() != '':
            loss_coeffs.append(float(new_loss_coeff))
            st.session_state.loss_coeffs = list(set(loss_coeffs))
            st.experimental_rerun()

    st.subheader('Embeddings')
    embeddings_path = st.text_input('Path', value='data/embeddings/ViT-B_32', key='embeddings_path_text_input')
    from_run_metadata_embeddings = st.checkbox('From run metadata', value=False, key='from_run_metadata_embeddings_checkbox')
    embeddings_uploader = st.file_uploader('Upload embeddings', key='embeddings_file_uploader')
    if embeddings_uploader is not None:
        # Save the uploaded file to a temporary location
        temp_file = NamedTemporaryFile(delete=False, suffix='.'+embeddings_uploader.name.split('.')[-1])
        temp_file.write(embeddings_uploader.read())

        # Get the full path of the saved file
        embeddings_path = temp_file.name

        # Close the temporary file
        temp_file.close()

    st.subheader('Row index')
    row_index_path = st.text_input('Path', value='data/embeddings/row_index', key='row_index_path_text_input')
    from_run_metadata_row_index = st.checkbox('From run metadata', value=False, key='from_run_metadata_row_index_checkbox')
    row_index_uploader = st.file_uploader('Upload row index', key='row_index_file_uploader')
    if row_index_uploader is not None:
        # Save the uploaded file to a temporary location
        temp_file = NamedTemporaryFile(delete=False, suffix='.'+row_index_uploader.name.split('.')[-1])
        temp_file.write(row_index_uploader.read())

        # Get the full path of the saved file
        row_index_path = temp_file.name

        # Close the temporary file
        temp_file.close()

    st.subheader('Run')
    run = st.text_input('Run', value='IGMP1')

    st.subheader('Run arguments')
    capture_stderr = st.checkbox('Capture_strerr', value=False, key='capture_stderr_checkbox')
    capture_stdout = st.checkbox('Capture_stdout', value=False, key='capture_stdout_checkbox')
    capture_hardware_metrics = st.checkbox('Capture_hardware_metrics', value=False, key='capture_hardware_metrics_checkbox')
    capture_traceback = st.checkbox('Capture traceback', value=False, key='capture_traceback_checkbox')

    st.subheader('From Annotation Matrix')
    annotation_matrix_path = st.text_input('Path', value='data/csv_files/annotations_cleaned', key='annotation_matrix_path_text_input')
    from_run_metadata_annotation_matrix = st.checkbox('From run metadata', value=False, key='from_run_metadata_annotation_matrix_checkbox')
    annotation_matrix_uploader = st.file_uploader('Upload annotation matrix', key='annotation_matrix_file_uploader')
    if annotation_matrix_uploader is not None:
        # Save the uploaded file to a temporary location
        temp_file = NamedTemporaryFile(delete=False, suffix='.'+annotation_matrix_uploader.name.split('.')[-1])
        temp_file.write(annotation_matrix_uploader.read())

        # Get the full path of the saved file
        annotation_matrix_path = temp_file.name

        # Close the temporary file
        temp_file.close()

    st.subheader('Checkpoint Path')
    checkpoint_path = st.text_input('Path', value='training/checkpoints/chkpt_epoch_2990', key='checkpoint_path_text_input')

    st.subheader('Clustering method')
    clustering_method = st.selectbox('Clustering method', ['DBSCAN', 'Spectral', 'Mean Shift', 'KDE', 'Gaussian Mixture'], index=0)
    
    if clustering_method.lower() == 'dbscan':
        epsilon = st.number_input('Epsilon', value=0.5, format="%.2f")
        min_samples = st.number_input('Min Samples', value=5)
        metric = st.selectbox('Metric', ['euclidean', 'manhattan', 'cosine'], index=0)
        algorithm = st.selectbox('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'], index=0)
        leaf_size = st.number_input('Leaf Size', value=30)
        p = st.number_input('P', value=None, format="%.2f")
        n_jobs = st.number_input('N Jobs', value=None)
        
    elif clustering_method.lower() == 'spectral':
        n_clusters = st.number_input('Number of Clusters', value=8)
        eigen_solver = st.selectbox('Eigen Solver', ['arpack', 'lobpcg', 'amg', None], index=0)
        n_components = st.number_input('Number of Components', value=None)
        random_state = st.number_input('Random State', value=None)
        n_init = st.number_input('Number of Initializations', value=10)
        gamma = st.number_input('Gamma', value=1.0)
        affinity = st.selectbox('Affinity', ['rbf', 'nearest_neighbors'], index=0)
        n_neighbors = st.number_input('Number of Neighbors', value=10)
        eigen_tol = st.number_input('Eigen Tolerance', value=None, format="%.2f")
        assign_labels = st.selectbox('Assign Labels', ['kmeans', 'discretize', 'cluster_qr'], index=0)
        degree = st.number_input('Degree', value=3)
        coef0 = st.number_input('Coef0', value=1.0)
        kernel_params = st.text_input('Kernel Parameters', value=None)
        n_jobs = st.number_input('Number of Jobs', value=None)

    elif clustering_method.lower().replace(" ", "_") == 'mean_shift':
        bandwidth = st.number_input('Bandwidth', value=None)
        seeds = st.text_area('Seeds', value=None)
        bin_seeding = st.checkbox('Bin Seeding', value=False)
        min_bin_freq = st.number_input('Minimum Bin Frequency', value=1)
        cluster_all = st.checkbox('Cluster All', value=True)
        n_jobs = st.number_input('Number of Jobs', value=None)
        max_iter = st.number_input('Maximum Iterations', value=300)

    elif clustering_method.lower() == 'kde':
        bandwidth = st.number_input('Bandwidth', value=1.0)
        algorithm = st.selectbox('Algorithm', ['kd_tree', 'ball_tree', 'auto'], index=2)
        kernel = st.selectbox('Kernel', ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'], index=0)
        metric = st.text_input('Metric', value='euclidean')
        atol = st.number_input('Absolute Tolerance', value=0.0, format="%.4f")
        rtol = st.number_input('Relative Tolerance', value=0.0, format="%.4f")
        breadth_first = st.checkbox('Breadth First', value=True)
        leaf_size = st.number_input('Leaf Size', value=40)
        metric_params = st.text_input('Metric Parameters', value=None)

    elif clustering_method.lower().replace(" ", "_") == 'gaussian_mixture':
        n_components = st.number_input('Number of Components', value=1)
        covariance_type = st.selectbox('Covariance Type', ['full', 'tied', 'diag', 'spherical'], index=0)
        tol = st.number_input('Tolerance', value=0.001)
        reg_covar = st.number_input('Regularization Covariance', value=0.000001)
        max_iter = st.number_input('Maximum Iterations', value=100)
        n_init = st.number_input('Number of Initializations', value=1)
        init_params = st.selectbox('Initialization Parameters', ['kmeans', 'k-means++', 'random', 'random_from_data'], index=0)
        random_state = st.number_input('Random State', value=None)
        warm_start = st.checkbox('Warm Start', value=False)


    submit_button_clicked = st.button('Submit')

    if submit_button_clicked or config is not None:
        # Validate the form
        submitted = True

        config_file = st.file_uploader("Upload JSON configuration file")
        if config_file is not None:
            # Save the uploaded file to a temporary location
            temp_file = NamedTemporaryFile(delete=False, suffix='.'+config_file.name.split('.')[-1])
            temp_file.write(config_file.read())

            # Get the full path of the saved file
            full_path = temp_file.name

            # Close the temporary file
            temp_file.close()

            st.write("Configuration file uploaded:", full_path)

        if config_file is not None:
            with open(full_path, "r") as f:
                config = json.load(f)
            os.remove(full_path)
        else:
            if clustering_method == 'DBSCAN':
                clustering_args = {
                    "eps": epsilon,
                    "min_samples": min_samples,
                    "metric": metric,
                    "algorithm": algorithm,
                    "leaf_size": leaf_size,
                    "p": p,
                    "n_jobs": n_jobs
                }
            elif clustering_method == 'Spectral':
                if eigen_tol == None:
                    eigen_tol = 'auto'
                clustering_args = {
                    "n_clusters": n_clusters,
                    "eigen_solver": eigen_solver,
                    "n_components": n_components,
                    "random_state": random_state,
                    "n_init": n_init,
                    "gamma": gamma,
                    "affinity": affinity,
                    "n_neighbors": n_neighbors,
                    "eigen_tol": eigen_tol,
                    "assign_labels": assign_labels,
                    "degree": degree,
                    "coef0": coef0,
                    "kernel_params": kernel_params,
                    "n_jobs": n_jobs
                }
                
            elif clustering_method == 'Mean Shift':
                clustering_args = {
                    "bandwidth": bandwidth,
                    "seeds": seeds,
                    "bin_seeding": bin_seeding,
                    "min_bin_freq": min_bin_freq,
                    "cluster_all": cluster_all,
                    "n_jobs": n_jobs,
                    "max_iter": max_iter
                }
            elif clustering_method == 'KDE':
                clustering_args = {
                    "bandwidth": bandwidth,
                    "algorithm": algorithm,
                    "kernel": kernel,
                    "metric": metric,
                    "atol": atol,
                    "rtol": rtol,
                    "breadth_first": breadth_first,
                    "leaf_size": leaf_size,
                    "metric_params": metric_params
                }
            elif clustering_method == 'Gaussian Mixture':
                clustering_method = 'gmm'
                clustering_args = {
                    "n_components": n_components,
                    "covariance_type": covariance_type,
                    "tol": tol,
                    "reg_covar": reg_covar,
                    "max_iter": max_iter,
                    "n_init": n_init,
                    "init_params": init_params,
                    "random_state": random_state,
                    "warm_start": warm_start
                }

            embeddings = {"path": embeddings_path,
                        "from_run_metadata": from_run_metadata_embeddings} if embeddings_uploader is None else embeddings_path
            
            row_index = {"path": row_index_path, "from_run_metadata": from_run_metadata_row_index} if row_index_uploader is None else row_index_path

            from_annotation_matrix = {"path": annotation_matrix_path, 
                                      "from_run_metadata": from_run_metadata_annotation_matrix} if annotation_matrix_uploader is None else annotation_matrix_path
            config = {
                "model_args": {
                    "layer_sizes": layer_sizes,
                    "loss": loss,
                    "loss_coeffs": loss_coeffs,
                    "loss_args": {}
                },
                "embeddings": embeddings,
                "row_index": row_index,
                "run": run,
                "run_args": {
                    "capture_stderr": capture_stderr,
                    "capture_stdout": capture_stdout,
                    "capture_hardware_metrics": capture_hardware_metrics,
                    "capture_traceback": capture_traceback
                },
                "from_annotation_matrix": from_annotation_matrix,
                "checkpoint_path": checkpoint_path,
                "clustering_method": clustering_method.lower().replace(" ", "_"),
                "clustering_args": clustering_args
            }
        if not use_custom_loss:
            loss = eval(f"nn.{config['model_args']['loss']}")
            config["model_args"]["loss"] = loss
        else:
            loss = eval(f"metrics.{config['model_args']['loss']}")
            config["model_args"]["loss"] = loss

    return config, submitted

def main():
    st.title('Evaluation')

    # Call the evaluation_form function to display the form
    option = st.sidebar.selectbox(
    'Select an option:',
    ('Training', 'Evaluation')
    )

    if option == 'Training':
        st.write('Training')
        st.session_state["dataframe"] = None
        st.session_state["config"] = None
    elif option == 'Evaluation':
        submitted = False
        dataframe = st.session_state.get('dataframe', None)
        config = st.session_state.get('config', None)

        with st.sidebar:
            
            st.session_state.config, submitted = evaluation_form(st.session_state.config)
        if st.session_state.dataframe is None and submitted and st.session_state.config is not None:
            st.session_state.dataframe = igmp_evaluator(**st.session_state.config)
        elif st.session_state.dataframe is not None and st.session_state.config is not None:
            data = clustering.clustering(method = st.session_state.config["clustering_method"], data = st.session_state.dataframe, **st.session_state.config["clustering_args"]).copy()
            clusters = data["cluster"].values
       
            pca = PCA(n_components=3)
            pca.fit(data.drop(columns=['cluster']))
            data_pca = pca.transform(data.drop(columns=['cluster']))
            data_viz = pd.DataFrame({'x': data_pca[:, 0], 'y': data_pca[:, 1], 'z': data_pca[:, 2], 'cluster': data['cluster']})
            fig = visualize.plot_clusters(data_viz, cluster_column='cluster', width=1000, height=800)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
