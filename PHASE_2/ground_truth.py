import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, ExpectationMaximization, BayesianEstimator
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, BDsScore
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
import networkx as nx

def visualise(model, pos, projectile_color="skyblue", interceptor_color="lightcoral"):
    edges = model.edges()

    # Create a directed graph object
    G = nx.DiGraph()

    # Add edges to the graph
    G.add_edges_from(edges)

    # Create a color map based on node type
    color_map = []
    for node in G.nodes():
        if "P_" in node:
            color_map.append(projectile_color)
        elif "I_" in node:
            color_map.append(interceptor_color)
        else:
            color_map.append("gray")  # Default color for any other nodes

    # Plot the graph with custom positions and colors
    plt.figure(figsize=(20, 12))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color=color_map, font_size=8, font_weight="bold", arrowsize=10)
    plt.title("Bayesian Network Graph")
    plt.show()

# Define positions manually
pos = {}

for i in range(6):

    if i % 2 == 0:
        # Projectile nodes positions (lower row)
        pos[f'P_FP{i}'] = ((i * 3)  , 0)
        pos[f'P_D{i}'] = ((i * 3 - 1)  , -1)
        pos[f'P_V{i}'] = ((i * 3 + 1)  , -1)
        pos[f'P_IP{i}'] = ((i * 3 - 1)  , -2)
        pos[f'P_A{i}'] = ((i * 3 + 1)  , -2)

        # Interceptor nodes positions (upper row)
        pos[f'I_FP{i}'] = (i * 3, 3)
        pos[f'I_D{i}'] = (i * 3 - 1, 2)
        pos[f'I_V{i}'] = (i * 3 + 1, 2)
        pos[f'I_IP{i}'] = (i * 3 - 1, 1)
        pos[f'I_A{i}'] = (i * 3 + 1, 1)
    else:
         # Projectile nodes positions (lower row)
        pos[f'P_FP{i}'] = ((i * 3)  , (0) - 1)
        pos[f'P_D{i}'] = ((i * 3 - 1)  ,( -1) - 1)
        pos[f'P_V{i}'] = ((i * 3 + 1)  , (-1) - 1)
        pos[f'P_IP{i}'] = ((i * 3 - 1)  , (-2) - 1)
        pos[f'P_A{i}'] = ((i * 3 + 1)  , (-2) - 1)

        # Interceptor nodes positions (upper row)
        pos[f'I_FP{i}'] = (i * 3, (3) + 1)
        pos[f'I_D{i}'] = (i * 3 - 1, (2) + 1)
        pos[f'I_V{i}'] = (i * 3 + 1, (2) + 1)
        pos[f'I_IP{i}'] = (i * 3 - 1, (1) + 1)
        pos[f'I_A{i}'] = (i * 3 + 1, (1) + 1)


# Create the Bayesian Network
GT = BayesianNetwork()

for i in range(6):
    GT.add_nodes_from([
        f'P_FP{i}', f'P_D{i}', f'P_V{i}',
        f'P_IP{i}', f'P_A{i}',
        f'I_FP{i}', f'I_D{i}', f'I_V{i}',
        f'I_IP{i}', f'I_A{i}'
    ])
    
    GT.add_edges_from([
        (f'P_FP{i}', f'P_D{i}'),
        (f'P_D{i}', f'P_IP{i}'),
        (f'P_D{i}', f'P_A{i}'),
        (f'P_FP{i}', f'P_V{i}'),
        (f'I_FP{i}', f'I_D{i}'),
        (f'I_D{i}', f'I_IP{i}'),
        (f'I_D{i}', f'I_A{i}'),
        (f'I_FP{i}', f'I_V{i}')
    ])

    if i < 5:
        # Connect corresponding nodes between time slices
        GT.add_edges_from([
            (f'P_FP{i}', f'P_FP{i + 1}'),
            (f'P_D{i}', f'P_D{i + 1}'),
            (f'P_V{i}', f'P_V{i + 1}'),
            (f'P_IP{i}', f'P_IP{i + 1}'),
            (f'P_A{i}', f'P_A{i + 1}'),
            (f'I_FP{i}', f'I_FP{i + 1}'),
            (f'I_D{i}', f'I_D{i + 1}'),
            (f'I_V{i}', f'I_V{i + 1}'),
            (f'I_IP{i}', f'I_IP{i + 1}'),
            (f'I_A{i}', f'I_A{i + 1}')
        ])
        
        # Edge between interceptor and projectile across time slices
        GT.add_edge(f'P_FP{i + 1}', f'I_FP{i}')

# Visualize the Bayesian Network with custom positions and colors
visualise(GT, pos)
