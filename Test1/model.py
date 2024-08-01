import pandas as pd
import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch
from pgmpy.models import DynamicBayesianNetwork as DBN




def define_dbn_structure():
    dbn = DBN()
    
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


scale_factor = 10
# Define the scaling factors (already scaled by 10 so max veloctity after this scale would be 2000 and angle would be 360)
velocity_scale_factor = 200
angle_scale_factor = 36

data = pd.DataFrame({
    ('ProjectileVelocity', 0): (np.random.rand(100) * scale_factor).astype(int),  # Scale up to some range, e.g., 0 to 100
    ('ProjectileAngle', 0): (np.random.rand(100) * scale_factor).astype(int),    # Angle range from 0 to 360 degrees
    ('InterceptorVelocity', 0): (np.random.rand(100) * scale_factor).astype(int),
    ('InterceptorAngle', 0): (np.random.rand(100) * scale_factor).astype(int),
    ('ProjectileVelocity', 1): (np.random.rand(100) * scale_factor).astype(int),
    ('ProjectileAngle', 1): (np.random.rand(100) * scale_factor).astype(int),
    ('InterceptorVelocity', 1): (np.random.rand(100) * scale_factor).astype(int),
    ('InterceptorAngle', 1): (np.random.rand(100) * scale_factor).astype(int),
})

# rounded_data = np.round(raw_data/10)*10
# data = rounded_data.astype(int)
print(data.head())



# Initialize the DBN structure
dbn = define_dbn_structure()

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

# Define evidence (current state) - scaled to the range 0 to 10
evidence = {
    ('ProjectileVelocity', 0): 1,
    ('ProjectileAngle', 0): 5,
}

# # Perform inference (predict the next state or optimal actions)
prediction = dbn_infer.forward_inference(variables=[('InterceptorVelocity', 1), ('InterceptorAngle', 1)], evidence=evidence)


# Extract and print the predicted values
predicted_values = {var: prediction[var].values for var in prediction}

# Reverse scaling function
def reverse_scale(value, is_angle=False):
    if is_angle:
        return value * angle_scale_factor
    else:
        return value * velocity_scale_factor

# Map the predicted discrete state back to continuous values
reverse_scaled_predictions = {
    var: reverse_scale(np.argmax(prediction[var].values), is_angle='Angle' in str(var)) for var in prediction
}

print("Predicted values (discrete):", predicted_values)
print("Reverse scaled predictions:", reverse_scaled_predictions)

