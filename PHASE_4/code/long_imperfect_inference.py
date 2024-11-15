# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, ExpectationMaximization, BayesianEstimator
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, BDsScore
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
import networkx as nx
import csv
from datetime import datetime
import time
import os

# %%

def visualise(model, pos, projectile_color="skyblue", interceptor_color="lightcoral"):
    edges = model.edges()

    # Create separate lists for intra-time slice and inter-time slice edges
    intra_time_slice_edges = []
    inter_time_slice_edges = []

    for edge in edges:
        if abs(int(edge[0][-1]) - int(edge[1][-1])) == 0:
            intra_time_slice_edges.append(edge)
        else:
            inter_time_slice_edges.append(edge)

    # Create a directed graph object
    G = nx.DiGraph()

    # Add edges to the graph
    G.add_edges_from(edges)

    # Create a color map based on node type
    color_map = []
    for node in G.nodes():
        if "FP" in node:
            color_map.append("gray")
        elif "P" == node[0]:
            color_map.append(projectile_color)
        elif "I_" in node:
            color_map.append(interceptor_color)
        else:
            color_map.append("gray")  # Default color for any other nodes

    # Plot the graph with custom positions and colors
    plt.figure(figsize=(20, 12))

    # Draw edges first to ensure arrowheads are not covered
    nx.draw_networkx_edges(
        G, pos, edgelist=intra_time_slice_edges, arrows=True,
        edge_color='black', style='solid', arrowsize=15, arrowstyle='-|>',
        min_source_margin=10, min_target_margin=15  # Increase margins to make arrows shorter
    )

    nx.draw_networkx_edges(
        G, pos, edgelist=inter_time_slice_edges, arrows=True,
        edge_color='black', style=(0, (5, 10)), arrowsize=15, arrowstyle='-|>',
        min_source_margin=10, min_target_margin=15  # Increase margins to make arrows shorter
    )

    # Draw nodes after edges
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=color_map)

    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

    plt.title("Bayesian Network Graph")
    plt.show()

# %%
# Define positions manually for x and y positions
pos = {}

for i in range(6):
    if i % 2 == 0:
        # Projectile nodes positions (lower row)
        pos[f'P_FP_x{i}'] = (i * 3 - 1 - 1, 0)          # Final position x
        pos[f'P_FP_y{i}'] = (i * 3 - 1, 0)              # Final position y
        pos[f'P_V{i}'] = (i * 3 + 1 - 1, -1)            # Velocity
        pos[f'P_IP_x{i}'] = (i * 3 - 1 - 1, -1)         # Initial position x
        pos[f'P_IP_y{i}'] = (i * 3 - 1, -1)             # Initial position y

        # Interceptor nodes positions (upper row)
        pos[f'I_FP_x{i}'] = (i * 3 - 1, 3)          # Final position x
        pos[f'I_FP_y{i}'] = (i * 3, 3)              # Final position y
        pos[f'I_V{i}'] = (i * 3 + 1, 2)             # Velocity
        pos[f'I_IP_x{i}'] = (i * 3 - 1, 2)          # Initial position x
        pos[f'I_IP_y{i}'] = (i * 3, 2)              # Initial position y
    else:
        # Projectile nodes positions (lower row)
        pos[f'P_FP_x{i}'] = (i * 3 - 1 - 1, -1)         # Final position x
        pos[f'P_FP_y{i}'] = (i * 3 - 1, -1)             # Final position y
        pos[f'P_V{i}'] = (i * 3 + 1 - 1, -2)            # Velocity
        pos[f'P_IP_x{i}'] = (i * 3 - 1 - 1, -2)         # Initial position x
        pos[f'P_IP_y{i}'] = (i * 3 - 1, -2)             # Initial position y

        # Interceptor nodes positions (upper row)
        pos[f'I_FP_x{i}'] = (i * 3 - 1, 4)          # Final position x
        pos[f'I_FP_y{i}'] = (i * 3, 4)              # Final position y
        pos[f'I_V{i}'] = (i * 3 + 1, 3)             # Velocity
        pos[f'I_IP_x{i}'] = (i * 3 - 1, 3)          # Initial position x
        pos[f'I_IP_y{i}'] = (i * 3, 3)              # Initial position y



# %%
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork

# Initialize the Bayesian Network model
GT = LinearGaussianBayesianNetwork()

for i in range(6):
    # Add nodes for final position (FP), velocity (V), and initial position (IP)
    GT.add_nodes_from([
        f'P_FP_x{i}', f'P_FP_y{i}', f'P_V{i}', f'P_IP_x{i}', f'P_IP_y{i}',  # Projectile nodes
        f'I_FP_x{i}', f'I_FP_y{i}', f'I_V{i}', f'I_IP_x{i}', f'I_IP_y{i}'   # Interceptor nodes
    ])
    
    # Add edges within each time slice for both x and y positions
    GT.add_edges_from([
        (f'P_FP_x{i}' , f'P_IP_x{i}'),  # Initial position (x) influences final position (x)
        (f'P_FP_x{i}' , f'P_V{i}'),     # Velocity influences final position (x)
        (f'P_FP_y{i}' , f'P_IP_y{i}'),  # Initial position (y) influences final position (y)
        (f'P_FP_y{i}', f'P_V{i}'),     # Velocity influences final position (y)
        
        (f'I_FP_x{i}' , f'I_IP_x{i}'),  # Interceptor: Initial position (x) influences final position (x)
        (f'I_FP_x{i}' , f'I_V{i}'),     # Velocity influences final position (x)
        (f'I_FP_y{i}' , f'I_IP_y{i}'),  # Initial position (y) influences final position (y)
        (f'I_FP_y{i}' , f'I_V{i}')      # Velocity influences final position (y)
    ])

    if i < 5:
        # Connect corresponding nodes between time slices for projectile and interceptor
        GT.add_edges_from([
            # Projectile time slice connections for x and y positions
            (f'P_FP_x{i}', f'P_FP_x{i + 1}'),
            (f'P_IP_x{i}', f'P_IP_x{i + 1}'),
            (f'P_FP_y{i}', f'P_FP_y{i + 1}'),
            (f'P_IP_y{i}', f'P_IP_y{i + 1}'),
            (f'P_V{i}', f'P_V{i + 1}'),

            # Interceptor time slice connections for x and y positions
            (f'I_FP_x{i}', f'I_FP_x{i + 1}'),
            (f'I_IP_x{i}', f'I_IP_x{i + 1}'),
            (f'I_FP_y{i}', f'I_FP_y{i + 1}'),
            (f'I_IP_y{i}', f'I_IP_y{i + 1}'),
            (f'I_V{i}', f'I_V{i + 1}')
        ])
        
        # Edge between projectile and interceptor across time slices
        GT.add_edges_from([
            (f'P_FP_x{i + 1}', f'I_FP_x{i}'),
            (f'P_FP_y{i + 1}', f'I_FP_y{i}')
        ])

# Final edges to ensure convergence at the last time slice
GT.add_edges_from([
    (f'P_FP_x{5}', f'I_FP_x{5}'),
    (f'P_FP_y{5}', f'I_FP_y{5}')
])


visualise(GT, pos)




# %%
def fit_cpd_for_node(node, parents, data):
    required_columns = [node] + parents

    # Ensure all required columns are in the data
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in data: {missing_columns}")
    
    # Creating the correct column names and structure for fitting
    node_data_before = data[required_columns].copy()
    node_data_before.columns = ["(Y|X)" if col == node else col for col in node_data_before.columns]

    # Add noise to all 0.0 values in the DataFrame
    noise_std = 1e-5 
    node_data = node_data_before.map(lambda x: x + np.random.normal(0, noise_std) if x == 0.0 else x)

    # print("Fitting CPD for node:", node)
    # print("Parents:", parents)
    # print("Data being used for fit:")
    # print(node_data.head())  # Print the first few rows of node_data

    if parents:
        # Initialize a Linear Gaussian CPD object
        cpd = LinearGaussianCPD(
            variable=node,
            evidence_mean=[0] * (len(parents) + 1),  # Initial guess for betas
            evidence_variance=1,  # Initial variance guess
            evidence=parents
        )
        
        node = "(Y|X)"
        # print(f"Node: {node} \n")
        # print("-"*50)
        # Fit the CPD using MLE
        beta, variance = cpd.fit(node_data, states=[node] + parents, estimator="MLE")
        return cpd
    else:
        # If no parents, fit a univariate Gaussian
        mean = data[node].mean()
        variance = data[node].var()
        return LinearGaussianCPD(variable=node, evidence_mean=[mean], evidence_variance=variance)


def predict_with_linear_gaussian_cpd(evidence, cpd, current_node):
    """
    Predict values using a Linear Gaussian CPD.
    
    Parameters
    ----------
    evidence: dict
        Dictionary with observed values.
    cpd: LinearGaussianCPD
        The CPD object to use for prediction.
    
    Returns
    -------
    float
        Predicted value for the query variable.
    """
    # print(cpd)

    if cpd is None:
        # If the CPD is None and the node is an FP node, estimate using IP and velocity
        if current_node.startswith('P_FP') or current_node.startswith('I_FP'):
            # Extract the relevant time slice from the node name
            time_slice = int(current_node.split('x')[-1]) if 'x' in current_node else int(current_node.split('y')[-1])

            if time_slice == 0:

                # Use the initial position and velocity to estimate final position
                ip_x = evidence.get(f'P_IP_x{time_slice}', 0) if current_node.startswith('P') else evidence.get(f'I_IP_x{time_slice}', 0)
                ip_y = evidence.get(f'P_IP_y{time_slice}', 0) if current_node.startswith('P') else evidence.get(f'I_IP_y{time_slice}', 0)
                ip_x = max(ip_x,0)
                ip_y = max(ip_y,0)
                velocity = evidence.get(f'P_V{time_slice}', 0) if current_node.startswith('P') else evidence.get(f'I_V{time_slice}', 0)
                if ip_x < 1500:
                    ip_x = 1500


                # Estimate the final position (this estimation can be modified based on the motion model)
                # predicted_value = (200 + time_slice*900 if 'x' in current_node else 100*time_slice + 200) if current_node.startswith('P') else (10000 - time_slice*900 if 'x' in current_node else 100*time_slice + 200)
                predicted_value = (1.2*ip_x if 'x' in current_node else 200) if current_node.startswith('P') else (0.95*ip_x if 'x' in current_node else 200)
            else:
                # Use the initial position, velocity and previous final to estimate final position
                ip_x = evidence.get(f'P_IP_x{time_slice}', 0) if current_node.startswith('P') else evidence.get(f'I_IP_x{time_slice}', 0)
                ip_y = evidence.get(f'P_IP_y{time_slice}', 0) if current_node.startswith('P') else evidence.get(f'I_IP_y{time_slice}', 0)
                ip_x = max(ip_x,0)
                ip_y = max(ip_y,0)
                velocity = evidence.get(f'P_V{time_slice}', 0) if current_node.startswith('P') else evidence.get(f'I_V{time_slice}', 0)
                fp_x = evidence.get(f'P_FP_x{time_slice - 1}', 0) if current_node.startswith('P') else evidence.get(f'I_FP_x{time_slice - 1}', 0)
                fp_y = evidence.get(f'P_FP_y{time_slice - 1}', 0) if current_node.startswith('P') else evidence.get(f'I_FP_y{time_slice - 1}', 0)
                ip_x = max(fp_x,0)
                ip_y = max(fp_y,0)

                # Estimate the final position (this estimation can be modified based on the motion model)
                # predicted_value = (200 + time_slice*900  if 'x' in current_node else 100*time_slice + 200) if current_node.startswith('P') else (10000 - time_slice*900 if 'x' in current_node else 100*time_slice + 200)
                predicted_value = (0.7*fp_x + 0.5*ip_x if 'x' in current_node else 1.25*fp_y) if current_node.startswith('P') else (0.9*fp_x if 'x' in current_node else 1.25*fp_y)

            # if current_node.startswith('P'):
            #     # print(f"Timeslice: {time_slice}")
            #     # print(f"ip_x = {ip_x}" if 'x' in current_node else f"ip_y = {ip_y}")
            #     # print(f"Estimate P = {predicted_value}")
            # else:
            #     # print(f"Timeslice: {time_slice}")
            #     # print(f"ip_x = {ip_x}" if 'x' in current_node else f"ip_y = {ip_y}")
            #     # print(f"Estimate I = {predicted_value}")

            return predicted_value
        else:
            # Return a fallback value if CPD is not fitted
            # print(f"CPD not fitted for node. Returning default value.")
            return 0.0  # Replace with domain-specific default if needed

    if not hasattr(cpd, 'beta') or cpd.beta is None:
        # The node has no parents, return its mean value
        # print(f"Node has no parents. Using mean for prediction.")
        return cpd.mean
    
    beta = cpd.beta
    mean_vector = cpd.mean
    variance = cpd.variance
    # print(f"Beta values: {beta}")
    # print(f"Mean vals: {mean_vector}")
    # print(f"Variance: {variance}")
    # print(f"Evidence required: {cpd.evidence}")
    evidence_vals = [evidence[var] for var in cpd.evidence]
    evidence_vals_array = np.array([val.flatten() if isinstance(val, np.ndarray) else [val] for val in evidence_vals])
    # print(f"Evidence vals: {evidence_vals_array}")
    # print(f"Coeffecients to be used: {beta[1:]}")
    predicted_mean = beta[0] + np.dot(
        evidence_vals_array.flatten(),
        beta[1:]
    )
    return predicted_mean




all_data = pd.read_csv('trajectory_data.csv')


# Fit the CPDs and measure time
start_train_time = time.time()
data = all_data


# Assuming you have a pandas DataFrame `data` with all necessary columns.
for i in range(6):
    if i == 0:
        # Fit Projectile CPDs for x and y positions
        # cpd_P_FP_x = fit_cpd_for_node(f'P_FP_x{i}', [], data)
        # cpd_P_FP_y = fit_cpd_for_node(f'P_FP_y{i}', [], data)
        cpd_P_V = fit_cpd_for_node(f'P_V{i}', [f'P_FP_x{i}', f'P_FP_y{i}'], data)
        cpd_P_IP_x = fit_cpd_for_node(f'P_IP_x{i}', [f'P_FP_x{i}'], data)
        cpd_P_IP_y = fit_cpd_for_node(f'P_IP_y{i}', [f'P_FP_y{i}'], data)
    else:
        # Fit Projectile CPDs for x and y positions
        # cpd_P_FP_x = fit_cpd_for_node(f'P_FP_x{i}', [f'P_FP_x{i - 1}'], data)
        # cpd_P_FP_y = fit_cpd_for_node(f'P_FP_y{i}', [f'P_FP_y{i - 1}'], data)
        cpd_P_V = fit_cpd_for_node(f'P_V{i}', [f'P_V{i - 1}',f'P_FP_x{i}', f'P_FP_y{i}'], data)
        cpd_P_IP_x = fit_cpd_for_node(f'P_IP_x{i}', [f'P_FP_x{i}', f'P_IP_x{i - 1}'], data)
        cpd_P_IP_y = fit_cpd_for_node(f'P_IP_y{i}', [f'P_FP_y{i}', f'P_IP_y{i - 1}'], data)

    # Add fitted CPDs to the network
    # GT.add_cpds(cpd_P_FP_x, cpd_P_FP_y, cpd_P_V, cpd_P_IP_x, cpd_P_IP_y)
    GT.add_cpds(cpd_P_V, cpd_P_IP_x, cpd_P_IP_y)

# Fit and add Interceptor CPDs
for i in range(6):
    if i == 0:
        # cpd_I_FP_x = fit_cpd_for_node(f'I_FP_x{i}', [f'P_FP_x{i+1}'], data)
        # cpd_I_FP_y = fit_cpd_for_node(f'I_FP_y{i}', [f'P_FP_y{i+1}'], data)
        cpd_I_V = fit_cpd_for_node(f'I_V{i}', [f'I_FP_x{i}', f'I_FP_y{i}'], data)
        cpd_I_IP_x = fit_cpd_for_node(f'I_IP_x{i}', [f'I_FP_x{i}'], data)
        cpd_I_IP_y = fit_cpd_for_node(f'I_IP_y{i}', [f'I_FP_y{i}'], data)
    else:
        # if i < 5:
            # cpd_I_FP_x = fit_cpd_for_node(f'I_FP_x{i}', [f'P_FP_x{i+1}', f'I_FP_x{i - 1}'], data)
            # cpd_I_FP_y = fit_cpd_for_node(f'I_FP_y{i}', [f'P_FP_y{i+1}', f'I_FP_y{i - 1}'], data)
        # else:
            # cpd_I_FP_x = fit_cpd_for_node(f'I_FP_x{i}', [f'I_FP_x{i - 1}', f'P_FP_x{i}'], data)
            # cpd_I_FP_y = fit_cpd_for_node(f'I_FP_y{i}', [f'I_FP_y{i - 1}', f'P_FP_y{i}'], data)

        cpd_I_V = fit_cpd_for_node(f'I_V{i}', [f'I_V{i - 1}',f'I_FP_x{i}',  f'I_FP_y{i}'], data)
        cpd_I_IP_x = fit_cpd_for_node(f'I_IP_x{i}', [f'I_FP_x{i}', f'I_IP_x{i - 1}'], data)
        cpd_I_IP_y = fit_cpd_for_node(f'I_IP_y{i}', [f'I_FP_y{i}', f'I_IP_y{i - 1}'], data)

    # Add fitted CPDs to the network
    # GT.add_cpds(cpd_I_FP_x, cpd_I_FP_y, cpd_I_V, cpd_I_IP_x, cpd_I_IP_y)
    GT.add_cpds(cpd_I_V, cpd_I_IP_x, cpd_I_IP_y)

end_train_time = time.time()
train_time = end_train_time - start_train_time


# %%
# Validation
# if GT.check_model():
#     print("The model is valid!")
# else:
#     print("The model is not valid. Please check the CPDs and network structure.")


# %%
import pandas as pd
import numpy as np

# Load the test data
all_test_data = pd.read_csv("trajectory_test_data.csv")


for i in range(10000, len(all_data) + 1, 10000):
    # Perform inference and measure time
    start_inference_time = time.time()
    test_data = all_test_data.iloc[:i]
    # List to store predictions for each row in test_data
    predictions = []

    # Iterate through each row in the test_data DataFrame
    for index, row in test_data.iterrows():
        # Define initial evidence (time slice 0)
        evidence = {
            'P_IP_x0' : row['P_IP_x0'],   
            'P_IP_y0' : row['P_IP_y0'],   
            'P_V0'  : row['P_V0'],
            'I_IP_x0' : row['I_IP_x0'],    
            'I_IP_y0' : row['I_IP_y0'],
            'I_V0':  row['I_V0'],
        }
        
        # Empty dict to store predictions for this row
        row_predictions = {}

        # COMPLETING PREDICTION FOR FIRST TREE

        cpd_x = GT.get_cpds(f'P_FP_x0')
        cpd_y = GT.get_cpds(f'P_FP_y0')
        
        prediction_x = predict_with_linear_gaussian_cpd(evidence, cpd_x, f'P_FP_x0')
        evidence[f'P_FP_x0'] = prediction_x

        prediction_y = predict_with_linear_gaussian_cpd(evidence, cpd_y, f'P_FP_y0')
        evidence[f'P_FP_y0'] = prediction_y
        
        row_predictions[f'P_FP_x0'] = prediction_x
        row_predictions[f'P_FP_y0'] = prediction_y
        
        # Propagate predictions for multiple time slices
        for t in range(1,6):
            # Predict for time slice t using evidence from time slice t-1
            cpd_v = GT.get_cpds(f'P_V{t}')
            cpd_I_x = GT.get_cpds(f'P_IP_x{t}')
            cpd_I_y = GT.get_cpds(f'P_IP_y{t}')
            cpd_x = GT.get_cpds(f'P_FP_x{t}')
            cpd_y = GT.get_cpds(f'P_FP_y{t}')

            prediction_x = predict_with_linear_gaussian_cpd(evidence, cpd_x, f'P_FP_x{t}')
            evidence[f'P_FP_x{t}'] = prediction_x

            prediction_y = predict_with_linear_gaussian_cpd(evidence, cpd_y, f'P_FP_y{t}')
            evidence[f'P_FP_y{t}'] = prediction_y
            
            prediction_v = predict_with_linear_gaussian_cpd(evidence, cpd_v, f'P_V{t}')
            evidence[f'P_V{t}'] = prediction_v

            prediction_I_x = predict_with_linear_gaussian_cpd(evidence, cpd_I_x, f'P_IP_x{t}')
            evidence[f'P_IP_x{t}'] = prediction_I_x

            prediction_I_y = predict_with_linear_gaussian_cpd(evidence, cpd_I_y, f'P_IP_y{t}')
            evidence[f'P_IP_y{t}'] = prediction_I_y

            
            # Store the predictions for this time slice
            row_predictions[f'P_V{t}'] = prediction_v
            row_predictions[f'P_IP_x{t}'] = prediction_I_x
            row_predictions[f'P_IP_y{t}'] = prediction_I_y
            row_predictions[f'P_FP_x{t}'] = prediction_x
            row_predictions[f'P_FP_y{t}'] = prediction_y

            #INTERCEPTOR NODES
            if t == 1:
                prediction_x = predict_with_linear_gaussian_cpd(evidence, cpd_x, f'I_FP_x{t - 1}')
                evidence[f'I_FP_x{t - 1}'] = prediction_x

                prediction_y = predict_with_linear_gaussian_cpd(evidence, cpd_y, f'I_FP_y{t - 1}')
                evidence[f'I_FP_y{t - 1}'] = prediction_y

                row_predictions[f'I_FP_x{t - 1}'] = prediction_x
                row_predictions[f'I_FP_y{t - 1}'] = prediction_y
            elif t > 1 and t < 5:

                # Predict for time slice t using evidence from time slice t-1
                cpd_v = GT.get_cpds(f'I_V{t - 1}')
                cpd_I_x = GT.get_cpds(f'I_IP_x{t - 1}')
                cpd_I_y = GT.get_cpds(f'I_IP_y{t - 1}')
                cpd_x = GT.get_cpds(f'I_FP_x{t - 1}')
                cpd_y = GT.get_cpds(f'I_FP_y{t - 1}')

                prediction_x = predict_with_linear_gaussian_cpd(evidence, cpd_x, f'I_FP_x{t - 1}')
                evidence[f'I_FP_x{t - 1}'] = prediction_x

                prediction_y = predict_with_linear_gaussian_cpd(evidence, cpd_y, f'I_FP_y{t - 1}')
                evidence[f'I_FP_y{t - 1}'] = prediction_y

                
                prediction_v = predict_with_linear_gaussian_cpd(evidence, cpd_v, f'I_V{t - 1}')
                evidence[f'I_V{t - 1}'] = prediction_v

                prediction_I_x = predict_with_linear_gaussian_cpd(evidence, cpd_I_x, f'I_IP_x{t - 1}')
                evidence[f'I_IP_x{t - 1}'] = prediction_I_x

                prediction_I_y = predict_with_linear_gaussian_cpd(evidence, cpd_I_y, f'I_IP_y{t - 1}')
                evidence[f'I_IP_y{t - 1}'] = prediction_I_y

                
                
                
                # Store the predictions for this time slice
                row_predictions[f'I_V{t - 1}'] = prediction_v
                row_predictions[f'I_IP_x{t - 1}'] = prediction_I_x
                row_predictions[f'I_IP_y{t - 1}'] = prediction_I_y
                row_predictions[f'I_FP_x{t - 1}'] = prediction_x
                row_predictions[f'I_FP_y{t - 1}'] = prediction_y
            
            else:
                # for the last set of interceptor nodes, when t = 5 then 4 is calculated since I've been using t-1, so for the last t, we do 2 sets of predictions for 4 and then 5
                for i in range (5, 7):
                    cpd_v = GT.get_cpds(f'I_V{i - 1}')
                    cpd_I_x = GT.get_cpds(f'I_IP_x{i - 1}')
                    cpd_I_y = GT.get_cpds(f'I_IP_y{i - 1}')
                    cpd_x = GT.get_cpds(f'I_FP_x{i - 1}')
                    cpd_y = GT.get_cpds(f'I_FP_y{i - 1}')

                    
                    prediction_x = predict_with_linear_gaussian_cpd(evidence, cpd_x, f'I_FP_x{i - 1}')
                    evidence[f'I_FP_x{i - 1}'] = prediction_x

                    prediction_y = predict_with_linear_gaussian_cpd(evidence, cpd_y, f'I_FP_y{i - 1}')
                    evidence[f'I_FP_y{i - 1}'] = prediction_y
                    
                    prediction_v = predict_with_linear_gaussian_cpd(evidence, cpd_v, f'I_V{i - 1}')
                    evidence[f'I_V{i - 1}'] = prediction_v

                    prediction_I_x = predict_with_linear_gaussian_cpd(evidence, cpd_I_x, f'I_IP_x{i - 1}')
                    evidence[f'I_IP_x{i - 1}'] = prediction_I_x

                    prediction_I_y = predict_with_linear_gaussian_cpd(evidence, cpd_I_y, f'I_IP_y{i - 1}')
                    evidence[f'I_IP_y{i - 1}'] = prediction_I_y


                    
                    
                    # Store the predictions for this time slice
                    row_predictions[f'I_V{i - 1}'] = prediction_v
                    row_predictions[f'I_IP_x{i - 1}'] = prediction_I_x
                    row_predictions[f'I_IP_y{i - 1}'] = prediction_I_y
                    row_predictions[f'I_FP_x{i - 1}'] = prediction_x
                    row_predictions[f'I_FP_y{i - 1}'] = prediction_y

        # Append predictions for this row
        predictions.append(row_predictions)

    # Perform inference as per your existing logic here...
    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time

    # Convert predictions to a DataFrame for easier analysis
    predictions_df = pd.DataFrame(predictions)

    # Print the prediction results
    # print(predictions_df)



    # %%
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    # Create lists to store the actual and predicted values for comparison
    actual_values = []
    predicted_values = []

    # Define the list of columns that we are interested in for comparison
    columns_to_compare = [f'P_V{t}' for t in range(1,6)] + \
                        [f'P_IP_x{t}' for t in range(1,6)] + \
                        [f'P_IP_y{t}' for t in range(1,6)] + \
                        [f'P_FP_x{t}' for t in range(6)] + \
                        [f'P_FP_y{t}' for t in range(6)] + \
                        [f'I_V{t}' for t in range(1,6)] + \
                        [f'I_IP_x{t}' for t in range(1,6)] + \
                        [f'I_IP_y{t}' for t in range(1,6)] + \
                        [f'I_FP_x{t}' for t in range(6)] + \
                        [f'I_FP_y{t}' for t in range(6)]

    # Iterate over each row in the test data
    for index, row in test_data.iterrows():
        actual_row = []
        predicted_row = []

        # Get actual values and predicted values for the current row
        for col in columns_to_compare:
            actual_value = row[col]
            predicted_value = predictions_df.loc[index, col]
            
            actual_row.append(actual_value)
            predicted_row.append(predicted_value)

            # Print the column name and the corresponding values
        #     print(f"Column: {col}")
        #     print(f"  Actual: {actual_value}")
        #     print(f"  Predicted: {predicted_value}")
        #     print("-" * 30)

        # # Print the row-wise summary
        # print("=" * 50 + f" ROW {index} SUMMARY " + "=" * 50)
        # print(f"Actual Values: {actual_row}")
        # print(f"Predicted Values: {predicted_row}")
        # print("\n")

        actual_values.extend(actual_row)
        predicted_values.extend(predicted_row)

    # Convert the lists to numpy arrays for metric calculation
    actual_values = np.array(actual_values)
    predicted_values = np.array([val.flatten() if isinstance(val, np.ndarray) else [val] for val in predicted_values])

    # Calculate accuracy metrics
    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)

    # Print the results
    # print(f"Mean Absolute Error (MAE): {mae}")
    # print(f"Mean Squared Error (MSE): {mse}")
    # print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Logging the results to CSV
    csv_file = 'imperfect_inference_model_performance_log.csv'
    header = ["Date", "Train Size", "Test Size", "MSE", "Train Time (s)", "Inference Time (s)"]

    # Check if the CSV file already exists
    write_header = not os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file doesn't exist
        if write_header:
            writer.writerow(header)
            
        # Write data row
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            len(data),
            len(test_data),
            mse,
            train_time,
            inference_time
        ])

    print("Results logged to", csv_file)


