In this phase, I aim to develop a basic network that will serve as the ground truth and so generate trajectories for training. I will then train a model with the generated trajectoy data and evaluate the results. I aim to do this by:

1) Creating the most basic network (1 dimensional). It will generate trajectories that resemble projectiles but the values do not need to be realistic. As long as some sort of meaningful interception can be generated. The following steps will take place:

    Step (1) Plot it better to make sure everything is connected properly
    Step (2) Sample parameter data
    Step (3) Forward sample
    Step (4) Generate data using forward sampling
    Step (5) Put data into correct format
    Step (6) Generate learned model

2) Then, I will make a more meaningful model by increasing it to 2 dimensions and follow similar steps.

The above phases have been completed successfully. In the next phase, I hope to improve on the 2-D ground truth model such that it produces trajectories that more accurately resemble projectile motion. All final files are in the ground truth folder names 2-D working