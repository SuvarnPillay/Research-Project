import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator,ExpectationMaximization, BayesianEstimator
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, BDsScore
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
import networkx as nx


def visualise(model):
    layout = 'planar'
    edges = model.edges()

    # Create a directed graph object
    G = nx.DiGraph()

    # Add edges to the graph
    G.add_edges_from(edges)

    # Choose the layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.5)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'planar':
        pos = nx.planar_layout(G)
    else:
        pos = nx.spring_layout(G)  # Default

    # Plot the graph
    plt.figure(figsize=(20, 12))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=12, font_weight="bold", arrowsize=10)
    plt.title("Bayesian Network Graph")
    plt.show()






Projectile_edges = []

Interceptor_nodes = []
Projectile_nodes = []

Interceptor_edges = []

for i in range(6):
    Projectile_nodes.append([f'Projectile_Final_Position_{i}', f'Projectile_Direction_{i}', f'Projectile_Velocity_{i}',
                              f'Projectile_Initial_Position_{i}',  f'Projectile_Angle_{i}'])
    
    Interceptor_nodes.append([f'Interceptor_Final_Position_{i}', f'Interceptor_Direction_{i}', f'Interceptor_Velocity_{i}',
                              f'Interceptor_Initial_Position_{i}', f'Interceptor_Angle_{i}'])
    
    Projectile_edges.append([(f'Projectile_Final_Position_{i}',f'Projectile_Direction_{i}'),(f'Projectile_Direction_{i}',f'Projectile_Initial_Position_{i}'),
                    ( f'Projectile_Direction_{i}',f'Projectile_Angle_{i}'),(f'Projectile_Final_Position_{i}',f'Projectile_Velocity_{i}')])
    
    Interceptor_edges.append([(f'Interceptor_Final_Position_{i}',f'Interceptor_Direction_{i}'),(f'Interceptor_Direction_{i}',f'Interceptor_Initial_Position_{i}'),
                    ( f'Interceptor_Direction_{i}',f'Interceptor_Angle_{i}'),(f'Interceptor_Final_Position_{i}',f'Interceptor_Velocity_{i}')])
                              
    
for i in range(5):
    Interceptor_edges.append([(f'Interceptor_Final_Position_{i}',f'Projectile_Final_Position_{i + 1}')])

GT = BayesianNetwork()
for i in range(6):
    GT.add_nodes_from(Projectile_nodes[i])
    GT.add_nodes_from(Interceptor_nodes[i])
    GT.add_edges_from(Projectile_edges[i])
    GT.add_edges_from(Interceptor_edges[i])

GT.add_edges_from(Interceptor_edges[6])  

# Call the function with your model and a different layout
visualise(GT)

# GT.add_edges_from(Interceptor_nodes)