This is my research
In this test folder, I am purely brainstorming and gaining foundational information in order to understand the problem better in order to properly solve it

Steps:

1) Generate basic trajectories using physics equation
2) Create model that learns to intercept the trajectory
    - Model input is point of the projectile
    - Model output is an action to move the interceptor
3) Think about the problem fully
    - How do I want to solve the problem?
        > Control a missile/drone to the projectile?
        > Create a trajectory that somehow simulates missile movement accurately and show interception?
            - Relaxed version could be 2 of the same trajectories colliding? Might just be simple math. May be a good idea to solve this first. Give me idea of the time issues and other problems in the situation
            


    - What is the pgm for?
    - Could I just use the kalman filter?
    - Should I implement both and a NN to compare all methods?
    
4) Create a relaxed version of the problem (Should be pretty easy to solve and implement)
5) Add complexities until the final version is complete

LIMITATIONS
- May need to remove the continuous structure of the data and discretize it. -> Lower accuracies should incur from this