import pandas as pd
import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch
from pgmpy.models import DynamicBayesianNetwork as DBN

def define_dbn_structure():
    dbn = DBN()
    
    # Temporal dependencies: Same variable at different time steps
    edges = [
        (('MissilePositionX', 0), ('MissilePositionX', 1)),
        (('MissilePositionY', 0), ('MissilePositionY', 1)),
        (('MissileVelocity', 0), ('MissileVelocity', 1)),
        (('MissileAngle', 0), ('MissileAngle', 1)),
        (('InterceptorPositionX', 0), ('InterceptorPositionX', 1)),
        (('InterceptorPositionY', 0), ('InterceptorPositionY', 1)),
        (('InterceptorVelocity', 0), ('InterceptorVelocity', 1)),
        (('InterceptorAngle', 0), ('InterceptorAngle', 1))
    ]
    dbn.add_edges_from(edges)
    
    # Dependencies between current state variables
    dependencies = [
        (('MissilePositionX', 0), ('InterceptorPositionX', 0)),
        (('MissilePositionY', 0), ('InterceptorPositionY', 0)),
        (('MissileVelocity', 0), ('InterceptorVelocity', 0)),
        (('MissileAngle', 0), ('InterceptorAngle', 0)),
    ]
    dbn.add_edges_from(dependencies)
    
    return dbn

# Example data preparation
# Replace these with actual data arrays from your data generator
missile_x_data_t0 = np.random.rand(100)  # Example data
missile_y_data_t0 = np.random.rand(100)
missile_velocity_data_t0 = np.random.rand(100)
missile_angle_data_t0 = np.random.rand(100)
interceptor_x_data_t0 = np.random.rand(100)
interceptor_y_data_t0 = np.random.rand(100)
interceptor_velocity_data_t0 = np.random.rand(100)
interceptor_angle_data_t0 = np.random.rand(100)

missile_x_data_t1 = np.random.rand(100)  # Example data
missile_y_data_t1 = np.random.rand(100)
missile_velocity_data_t1 = np.random.rand(100)
missile_angle_data_t1 = np.random.rand(100)
interceptor_x_data_t1 = np.random.rand(100)
interceptor_y_data_t1 = np.random.rand(100)
interceptor_velocity_data_t1 = np.random.rand(100)
interceptor_angle_data_t1 = np.random.rand(100)

# Create the DataFrame with your data
data = pd.DataFrame({
    ('MissilePositionX', 0): missile_x_data_t0,
    ('MissilePositionY', 0): missile_y_data_t0,
    ('MissileVelocity', 0): missile_velocity_data_t0,
    ('MissileAngle', 0): missile_angle_data_t0,
    ('InterceptorPositionX', 0): interceptor_x_data_t0,
    ('InterceptorPositionY', 0): interceptor_y_data_t0,
    ('InterceptorVelocity', 0): interceptor_velocity_data_t0,
    ('InterceptorAngle', 0): interceptor_angle_data_t0,
    ('MissilePositionX', 1): missile_x_data_t1,
    ('MissilePositionY', 1): missile_y_data_t1,
    ('MissileVelocity', 1): missile_velocity_data_t1,
    ('MissileAngle', 1): missile_angle_data_t1,
    ('InterceptorPositionX', 1): interceptor_x_data_t1,
    ('InterceptorPositionY', 1): interceptor_y_data_t1,
    ('InterceptorVelocity', 1): interceptor_velocity_data_t1,
    ('InterceptorAngle', 1): interceptor_angle_data_t1,
})

# Initialize the DBN structure
dbn = define_dbn_structure()

# Fit the DBN using Maximum Likelihood Estimation
dbn.fit(data, estimator=MaximumLikelihoodEstimator)

# Define evidence for prediction
# Example current state values for evidence
current_missile_x = 1000  # Replace with actual current values
current_missile_y = 500
current_missile_velocity = 200
current_missile_angle = 45

evidence = {
    ('MissilePositionX', 0): current_missile_x,
    ('MissilePositionY', 0): current_missile_y,
    ('MissileVelocity', 0): current_missile_velocity,
    ('MissileAngle', 0): current_missile_angle
}

# Perform inference to predict interceptor's trajectory
# Predict positions and other attributes for the next time step
prediction = dbn.predict([('InterceptorPositionX', 1), ('InterceptorPositionY', 1),
                          ('InterceptorVelocity', 1), ('InterceptorAngle', 1)], evidence=evidence)

print(prediction)
