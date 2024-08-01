import pandas as pd
import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch
from pgmpy.models import DynamicBayesianNetwork as DBN




def define_dbn_structure():
    dbn = DBN()
    
    # Define nodes for time slices 0 and 1
    # nodes_t0 = ['ProjectileVelocity', 'ProjectileAngle', 'InterceptorVelocity', 'InterceptorAngle']
    # nodes_t1 = [f'{node}_1' for node in nodes_t0]
    
    # # Add nodes for time slice 0 and 1
    # for node in nodes_t0:
    #     dbn.add_node((node, 0))
    # for node in nodes_t1:
    #     dbn.add_node((node, 1))
    
    # Add edges for temporal dependencies
    edges = [
        (('ProjectileVelocity', 0), ('ProjectileVelocity', 1)),
        (('ProjectileAngle', 0), ('ProjectileAngle', 1)),
        (('InterceptorVelocity', 0), ('InterceptorVelocity', 1)),
        (('InterceptorAngle', 0), ('InterceptorAngle', 1))
    ]
    dbn.add_edges_from(edges)
    
    # Add edges for dependencies between current state variables
    dependencies = [
        (('ProjectileVelocity', 0), ('InterceptorVelocity', 0)),
        (('ProjectileAngle', 0), ('InterceptorAngle', 0))
    ]
    dbn.add_edges_from(dependencies)
    
    return dbn





# Create a DataFrame with columns named using time slices 0 and 1
data = pd.DataFrame({
    ('ProjectileVelocity', 0): np.random.rand(100),
    ('ProjectileAngle', 0): np.random.rand(100),
    ('InterceptorVelocity', 0): np.random.rand(100),
    ('InterceptorAngle', 0): np.random.rand(100),
    ('ProjectileVelocity', 1): np.random.rand(100),
    ('ProjectileAngle', 1): np.random.rand(100),
    ('InterceptorVelocity', 1): np.random.rand(100),
    ('InterceptorAngle', 1): np.random.rand(100),
})
print(data.head())


# Initialize the DBN structure
dbn = define_dbn_structure()

# # Use Hill Climb Search for structure learning
# hc_search = HillClimbSearch(data)
# best_model = hc_search.estimate()

# # Initialize the DBN with the learned structure
# dbn = DBN(best_model.edges())

# Add the learned structure to the DBN
# for edge in best_model.edges():
#     dbn.add_edge(edge[0], edge[1])

# Add nodes
# for node in data.columns:
#     if node.endswith('_1'):
#         dbn.add_node((node[:-2], 1))
#     else:
#         dbn.add_node((node, 0))

# Print the learned structure
print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Listing the nodes:",list(dbn.nodes()))
print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Listing the edges:",list(dbn.edges()))
print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Column names:", data.columns)
print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------")

# Fit the DBN model with data using Maximum Likelihood Estimator
dbn.fit(data)



from pgmpy.inference import DBNInference

# # Create an inference object
dbn_infer = DBNInference(dbn)

# # Define evidence (current state)
evidence = {
    ('ProjectileVelocity', 0): 100,
    ('ProjectileAngle', 0): 45,
}

# # Perform inference (predict the next state or optimal actions)
prediction = dbn_infer.map_query(variables=[('InterceptorVelocity', 1), ('InterceptorAngle', 1)], evidence=evidence)
print(prediction)
