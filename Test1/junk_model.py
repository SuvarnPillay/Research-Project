from pgmpy.models import DynamicBayesianNetwork as DBN
import numpy as np
import pandas as pd

# Initialize the DBN
dbn = DBN()

# Define nodes for time slices 0 and 1 (time slices start from 0 as per documentation)
nodes_t0 = ['ProjectileVelocity', 'ProjectileAngle', 'InterceptorVelocity', 'InterceptorAngle']
nodes_t1 = [f'{node}_1' for node in nodes_t0]

# Add nodes for time slice 0 and 1
for node in nodes_t0:
    dbn.add_node((node, 0))
for node in nodes_t1:
    dbn.add_node((node, 1))

print(list(dbn.nodes()))
print(dbn.number_of_nodes())

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


# Load or create your data
data = pd.DataFrame({
    'ProjectileVelocity': np.random.rand(100),
    'ProjectileAngle': np.random.rand(100),
    'InterceptorVelocity': np.random.rand(100),
    'InterceptorAngle': np.random.rand(100),
    'ProjectileVelocity_1': np.random.rand(100),
    'ProjectileAngle_1': np.random.rand(100),
    'InterceptorVelocity_1': np.random.rand(100),
    'InterceptorAngle_1': np.random.rand(100),
})

# Check column names to ensure they start from time slice 0
# print(data.columns)


# Fit the DBN model with data using Maximum Likelihood Estimator by default
dbn.fit(data)



from pgmpy.inference import DBNInference

# Create an inference object
dbn_infer = DBNInference(dbn)

# Define evidence (current state)
evidence = {
    ('ProjectileVelocity', 0): 100,
    ('ProjectileAngle', 0): 45,
}

# Perform inference (predict the next state or optimal actions)
prediction = dbn_infer.map_query(variables=[('InterceptorVelocity', 1), ('InterceptorAngle', 1)], evidence=evidence)
print(prediction)
