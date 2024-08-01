CONFIG = {
    'GUI': True,  # Enable GUI mode for the simulation
    'create_model': False,  # Flag to create a new model or use an existing model
    'ACTION_LIST': [(-1.57, -20), (-1.57, 0), (-1.57, 20), (-1.57, 40), (-1.57, 60),
                    (-0.79, -20), (-0.79, 0), (-0.79, 20), (-0.79, 40), (-0.79, 60),
                    (0, -20), (0, 0), (0, 20), (0, 40), (0, 60),
                    (0.79, -20), (0.79, 0), (0.79, 20), (0.79, 40), (0.79, 60),
                    (1.57, -20), (1.57, 0), (1.57, 20), (1.57, 40), (1.57, 60)],  # List of possible actions (steering angle, velocity)
    'steer_step': 90,  # Step size for steering angles
    'weight_file_name': 'model/model_weight.pth',  # File name for saving/loading model weights
    'lr': 0.001,  # Learning rate for the optimizer
    'target_reward': 3000,  # Target reward value for termination
    'epochs': 3000,  # Number of epochs to train
    'episodes': 10,  # Number of episodes per epoch
    'collision_reward': -0.8,  # Reward for collision
    'timeover_reward': -0.2,  # Reward for timeout
    'max_number_of_steps': 18000,  # Maximum number of steps per episode
    'steeringLinks': [4, 6],  # List of joint indices for steering
    'wheelLinks': [2, 3, 5, 7]  # List of joint indices for wheels
}