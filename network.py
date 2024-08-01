import torch  
from torch import tensor, nn, optim  
import torch.nn.functional as F 
import os  
import numpy as np

class NeuralNetwork(nn.Module):
    '''
    NeuralNetwork class defines a simple feed-forward neural network with three hidden layers.
    The network uses ReLU activations for hidden layers and softmax activation for the output layer.

    Attributes:
        seq (torch.nn.Sequential): Sequential model containing the layers of the neural network.
    '''
    def __init__(self, dim_in, dim_out):
        '''
        Initialize the neural network with input and output dimensions.
        Create a sequential model with three hidden layers and ReLU activations.

        Args:
            dim_in (int): Dimension of the input layer.
            dim_out (int): Dimension of the output layer.
        '''
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim_in, 64),  # First hidden layer with 64 nodes
            nn.ReLU(),  # ReLU activation
            nn.Linear(64, 32),  # Second hidden layer with 32 nodes
            nn.ReLU(),  # ReLU activation
            nn.Linear(32, 32),  # Third hidden layer with 32 nodes
            nn.ReLU(),  # ReLU activation
            nn.Linear(32, dim_out)  # Output layer
        )

    def forward(self, x):
        '''
        Forward pass of the neural network.
        Apply the sequential model and return the softmax of the output.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with softmax activation.
        '''
        return F.softmax(self.seq(x), dim=0)

def setup_network(dim_in, dim_out, create_model, device, weight_file_name, lr):
    '''
    Setup the neural network model and optimizer.
    If create_model is True or weight file does not exist, create a new model.
    Otherwise, load the pre-trained model from the weight file.

    Args:
        dim_in (int): Dimension of the input layer.
        dim_out (int): Dimension of the output layer.
        create_model (bool): Flag indicating whether to create a new model.
        device (torch.device): Device to load the model onto (CPU or GPU).
        weight_file_name (str): Name of the file containing the model weights.
        lr (float): Learning rate for the optimizer.

    Returns:
        tuple: A tuple containing the model and the optimizer.
    '''
    if create_model or not os.path.isfile(weight_file_name):
        model = NeuralNetwork(dim_in=dim_in, dim_out=dim_out).to(device)  # Create a new model
    else:
        model = torch.load(weight_file_name).to(device)  # Load pre-trained model

    optimizer = optim.Adam(model.parameters(), lr=lr)  # Create an Adam optimizer
    return model, optimizer

def update_policy(rewards, policies, steps, opt):
    '''
    Update the policy using policy gradients.
    Calculate the average reward and update the model parameters using backpropagation.

    Args:
        rewards (torch.Tensor): Tensor of rewards.
        policies (torch.Tensor): Tensor of policy probabilities.
        steps (torch.Tensor): Tensor of steps.
        opt (torch.optim.Optimizer): Optimizer to update the model parameters.

    Returns:
        None
    '''
    reward_ave = (rewards.nansum(dim=1) / steps).mean()  # Average reward
    clampped = torch.clamp(policies, 1e-10, 1)  # Avoid log(0)
    Jmt = clampped.log() * (rewards - reward_ave)  # Calculate Jmt
    J = (Jmt.nansum(dim=1) / steps).mean()  # Average J
    J.backward()  # Backpropagation
    opt.step()  # Update optimizer
    opt.zero_grad()  # Reset optimizer

def decide_action(output, ACTION_LIST, steer_step):
    '''
    Decide the action to take based on the neural network output.
    Use a probability distribution to select the action.

    Args:
        output (torch.Tensor): Output tensor from the neural network.
        ACTION_LIST (list): List of possible actions.
        steer_step (int): Steering step in degrees.

    Returns:
        tuple: A tuple containing the action dictionary and the one-hot encoded action.
    '''
    prop = output.detach().numpy()  # Detach output and convert to numpy array
    one_hot = torch.zeros([len(ACTION_LIST)])  # One-hot tensor for actions
    action = np.random.choice(range(len(ACTION_LIST)), p=prop)  # Select action based on probability
    one_hot[action] = 1  # Set the chosen action in one-hot tensor

    steer, vel = ACTION_LIST[action]  # Get action (steer and velocity)
    steer *= np.deg2rad(steer_step)  # Convert steer to radians
    return {"steer": np.round(steer, decimals=2), "vel": np.round(vel, decimals=2)}, one_hot

def set_reward(distances, rv, angle_vel):
    '''
    Set the reward based on distances from LiDAR, wheel speed, and angular velocity.
    Reward is higher for greater distances from obstacles and lower for high angular velocity.

    Args:
        distances (list): List of distances from LiDAR.
        rv (float): Wheel speed.
        angle_vel (float): Angular velocity.

    Returns:
        float: Calculated reward.
    '''
    frontside = distances[int(len(distances)/4):int(3*len(distances)/4)]  # Frontside distances
    leftside = distances[:int(len(distances)/2)]  # Leftside distances
    rightside = distances[int(len(distances)/2):]  # Rightside distances

    reward = 0  # Initialize reward
    # Reward based on distance to obstacles
    for dist in frontside:
        if dist > 2.0:
            reward += dist * 1.5
        elif dist < 2.0:
            reward -= 1 / dist

    for dist in leftside:
        if dist < 0.5:
            reward -= 1 / dist / len(leftside) * 0.5

    for dist in rightside:
        if dist < 0.5:
            reward -= 1 / dist / len(rightside) * 0.5

    front = frontside[int(len(frontside)/2)]  # Front distance
    left = leftside[int(len(leftside)/2)]  # Left distance
    right = rightside[int(len(rightside)/2)]  # Right distance

    if front > 2.0:
        reward += front * 0.8
    if left > 2.0:
        reward -= left * 1.2
    if right > 2.0:
        reward -= right * 1.2

    reward += rv * 0.3  # Reward based on wheel speed
    if rv == 0:
        reward -= 0.5

    if abs(angle_vel) > 10:  # Prevent sharp steering
        reward -= 3.0

    return reward
