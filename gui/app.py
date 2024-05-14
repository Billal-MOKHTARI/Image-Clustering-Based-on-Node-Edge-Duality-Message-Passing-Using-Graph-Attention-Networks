import streamlit as st
from tempfile import NamedTemporaryFile
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.training.evaluator import igmp_evaluator
from models.networks import metrics
import json 
from torch import nn
from src import visualize
from sklearn.decomposition import PCA
import pandas as pd
st.set_page_config(layout="wide")

def evaluation_form():
    dataframe = None
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

    st.subheader('Row index')
    row_index_path = st.text_input('Path', value='data/embeddings/row_index', key='row_index_path_text_input')
    from_run_metadata_row_index = st.checkbox('From run metadata', value=False, key='from_run_metadata_row_index_checkbox')

    st.subheader('Run')
    run = st.text_input('Run', value='IGMP1')

    st.subheader('Run arguments')
    capture_stderr = st.checkbox('Capture_strerr', value=False, key='capture_stderr_checkbox')
    capture_stdout = st.checkbox('Capture_stdout', value=False, key='capture_stdout_checkbox')
    capture_hardware_metrics = st.checkbox('Capture_hardware_metrics', value=False, key='capture_hardware_metrics_checkbox')
    capture_traceback = st.checkbox('Capture traceback', value=False, key='capture_traceback_checkbox')

    st.subheader('From Annotation Matrix')
    annotation_matrix_path = st.text_input('Path', value='benchmark/datasets/agadez/csv_files/annotations_cleaned.csv', key='annotation_matrix_path_text_input')

    st.subheader('Checkpoint Path')
    checkpoint_path = st.text_input('Path', value='training/checkpoints/chkpt_epoch_2990', key='checkpoint_path_text_input')

    st.subheader('Clustering method')
    clustering_method = st.selectbox('Clustering method', ['DBSCAN'], index=0)
    if clustering_method.lower() == 'dbscan':
        epsilon = st.number_input('Epsilon', value=3.5, format="%.2f")
        min_samples = st.number_input('Min Samples', value=8)

    submit_button_clicked = st.button('Submit')

    if submit_button_clicked:
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
            config = {
                "model_args": {
                    "layer_sizes": layer_sizes,
                    "loss": loss,
                    "loss_coeffs": loss_coeffs,
                    "loss_args": {}
                },
                "embeddings": {
                    "path": embeddings_path,
                    "from_run_metadata": from_run_metadata_embeddings
                },
                "row_index": {
                    "path": row_index_path,
                    "from_run_metadata": from_run_metadata_row_index
                },
                "run": run,
                "run_args": {
                    "capture_stderr": capture_stderr,
                    "capture_stdout": capture_stdout,
                    "capture_hardware_metrics": capture_hardware_metrics,
                    "capture_traceback": capture_traceback
                },
                "from_annotation_matrix": annotation_matrix_path,
                "checkpoint_path": checkpoint_path,
                "clustering_method": clustering_method.lower(),
                "clustering_args": {
                    "eps": epsilon,
                    "min_samples": min_samples
                }
            }
        if not use_custom_loss:
            loss = eval(f"nn.{config['model_args']['loss']}")
            config["model_args"]["loss"] = loss
        else:
            loss = eval(f"metrics.{config['model_args']['loss']}")
            config["model_args"]["loss"] = loss
        
        dataframe = igmp_evaluator(**config)

    if submitted:
        st.write("Form submitted successfully!")

    return dataframe

def main():
    st.title('Evaluation')

    # Call the evaluation_form function to display the form
    option = st.sidebar.selectbox(
    'Select an option:',
    ('Training', 'Evaluation')
    )

    if option == 'Training':
        st.write('Training')
    elif option == 'Evaluation':
        dataframe = None
        with st.sidebar:
            dataframe = evaluation_form()

        if dataframe is not None:
            clusters = dataframe["cluster"].values

        if dataframe is not None:
            pca = PCA(n_components=3)
            pca.fit(dataframe.drop(columns=['cluster']))
            data_pca = pca.transform(dataframe.drop(columns=['cluster']))
            data_viz = pd.DataFrame({'x': data_pca[:, 0], 'y': data_pca[:, 1], 'z': data_pca[:, 2], 'cluster': dataframe['cluster']})
            fig = visualize.plot_clusters(data_viz, cluster_column='cluster', width=1000, height=800)
            st.plotly_chart(fig, use_container_width=True)
     # Check if the instructions have already been executed
    if 'instructions_executed' not in st.session_state:
        # Execute the instructions only once
        st.write("These instructions will be executed only once!")
        
        # Set a flag to indicate that the instructions have been executed
        st.session_state.instructions_executed = True


if __name__ == "__main__":
    main()
