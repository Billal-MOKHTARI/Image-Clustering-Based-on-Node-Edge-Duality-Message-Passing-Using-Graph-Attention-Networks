# import streamlit as st
# import subprocess
# import argparse
# import json
# import os
# import sys
# from torch import nn
# import torch
# import time

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# from models.training.evaluator import igmp_evaluator
# from models.networks import metrics

# os.system("sudo sysctl fs.inotify.max_user_watches=588842")
# config_path = "configs/igmp_evaluator/igmp_evaluator_1.json"
 

# with open(config_path, "r") as f:
#     config = json.load(f)

# use_custom_loss = config.get("use_custom_loss", False)

# if "model_args" in config.keys():
#     if not use_custom_loss:
#         loss = eval(f"nn.{config['model_args']['loss']}")
#         config["model_args"]["loss"] = loss
#     else:
#         loss = eval(f"metrics.{config['model_args']['loss']}")
#         config["model_args"]["loss"] = loss


# data, fig = igmp_evaluator(**config)


# with st.form(key='Clustering'):
#     eps = st.number_input(label = 'Epsilon')
#     min_samples = int(st.number_input(label = 'Min samples'))
#     st.plotly_chart(fig)
#     submitted = st.form_submit_button('Submit')
#     if submitted:
#         config["clustering_args"]["eps"] = eps
#         config["clustering_args"]["min_samples"] = min_samples
#         data, fig = igmp_evaluator(**config)
        
