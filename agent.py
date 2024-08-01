import pybullet as p
import lidar
import numpy as np
import torch
from torch import tensor, device, cuda
import matplotlib.pyplot as plt

from settings import setup_simulation, get_initial_position, setup_lidar
from control import control_steering, control_velocity, set_camera, end_episode
from network import setup_network, update_policy, decide_action, set_reward
from config import CONFIG
from recorder import save_model, save_graph, save_data

class ReinforcementLearningAgent:
    """A reinforcement learning agent for controlling a racecar in a PyBullet simulation environment.
    
    Attributes:
        physicsClient: The physics client for the PyBullet simulation.
        racecar: The racecar model in the simulation.
        simpleMap: The map used in the simulation.
        steeringLinks: Links for the steering control.
        wheelLinks: Links for the wheel control.
        hokuyo_joint: The joint for the lidar setup.
        model: The neural network model.
        optimizer: The optimizer for the neural network.
        t: Time step counter.
        step_interval: Interval between steps.
        episode: Episode counter.
        step: Step counter.
        epoch: Epoch counter.
        experiences: List of experiences in the current episode.
        rewards: Tensor of rewards for the episodes.
        pre_rewards: Tensor of rewards for the previous episodes.
        policies: Tensor of policies for the episodes.
        steps: Tensor of steps for the episodes.
        policy_list: List of policies for the current episode.
        reward_list: List of rewards for the current episode.
        mean_reward_list: List of mean rewards.
        posPrev: Previous position of the racecar.
        pos: Current position of the racecar.
        graph_x: X values for the graph.
        graph_y: Y values for the graph.
        record_rewards: List of recorded rewards.
    """

    def __init__(self):
        """Initializes the reinforcement learning agent, sets up the simulation environment and the neural network."""
        self.physicsClient, self.racecar, self.simpleMap, self.steeringLinks, self.wheelLinks = setup_simulation(GUI=CONFIG['GUI'])
        self.hokuyo_joint = setup_lidar(self.racecar)

        # Neural Network
        NUM_ACTIONS = len(CONFIG['ACTION_LIST'])
        NUM_STATE = lidar.numRays + 1
        self.device_type = device('cuda' if cuda.is_available() else 'cpu')
        self.model, self.optimizer = setup_network(NUM_STATE, NUM_ACTIONS, CONFIG['create_model'], self.device_type, CONFIG['weight_file_name'], CONFIG['lr'])

        # Variables
        self.t = 0  # Initialize time step counter
        self.step_interval = 0  # Initialize interval between steps
        self.episode = 0  # Initialize episode counter
        self.step = 0  # Initialize step counter
        self.epoch = 0  # Initialize epoch counter
        self.experiences = []  # Initialize list of experiences
        self.rewards = tensor([])  # Initialize tensor for rewards
        self.pre_rewards = tensor([])  # Initialize tensor for previous rewards
        self.policies = tensor([])  # Initialize tensor for policies
        self.steps = tensor([])  # Initialize tensor for steps
        self.policy_list = tensor([np.nan] * CONFIG['max_number_of_steps'])  # Initialize list of policies for the current episode
        self.reward_list = tensor([np.nan] * CONFIG['max_number_of_steps'])  # Initialize list of rewards for the current episode
        self.mean_reward_list = np.zeros((0))  # Initialize list of mean rewards
        self.posPrev = 0  # Initialize previous position of the racecar

        # Graph
        plt.figure()  # Create a new figure for plotting
        self.graph_x = []  # Initialize X values for the graph
        self.graph_y = []  # Initialize Y values for the graph

        # Excel
        self.record_rewards = []  # Initialize list of recorded rewards

        # Get initial position of the car model
        self.pos = get_initial_position(self.racecar)

    def process_step(self):
        """Processes a single step in the simulation, including deciding the action, controlling the racecar, and calculating rewards."""
        self.posPrev = self.pos  # Update previous position
        self.pos = get_initial_position(self.racecar)  # Get current position
        dx = np.sqrt((self.pos[0] - self.posPrev[0]) ** 2 + (self.pos[1] - self.posPrev[1]) ** 2)  # Calculate distance traveled

        # Input for Neural Network
        distances = [np.round(n, decimals=2) for n in lidar.detection(self.racecar, self.hokuyo_joint)]  # Get lidar distances
        v = np.round(dx / 0.1, decimals=2)  # Calculate velocity
        distances = torch.tensor(distances).float()  # Convert distances to tensor
        v = torch.tensor([v]).float()  # Convert velocity to tensor
        input = torch.cat([distances, v])  # Concatenate distances and velocity into input tensor

        # Decide action
        output = self.model(input)  # Get model output
        action, one_hot = decide_action(output, CONFIG['ACTION_LIST'], CONFIG['steer_step'])  # Decide action and get one-hot encoding

        _, rv, _, _ = [np.round(n, decimals=2) for n in p.getJointState(self.racecar, 2)]  # Get joint state
        qua = p.getBasePositionAndOrientation(self.racecar)[1]  # Get base orientation
        euler_z = p.getEulerFromQuaternion(qua)[2]  # Convert quaternion to Euler angles
        angle_vel = euler_z / 0.1  # Calculate angular velocity

        reward = set_reward(distances, rv, angle_vel)  # Calculate reward

        control_velocity(self.racecar, action["vel"], self.wheelLinks)  # Control racecar velocity
        control_steering(self.racecar, action["steer"], self.steeringLinks)  # Control racecar steering

        contact = p.getContactPoints(self.racecar, self.simpleMap)  # Check for contact points
        if contact or len(self.experiences) >= CONFIG['max_number_of_steps'] - 1:  # Check for collision or max steps
            self.racecar = end_episode(self.racecar)  # End the episode
            reward += CONFIG['collision_reward'] if contact else CONFIG['timeover_reward']  # Update reward
            self.policies = torch.cat([self.policies, self.policy_list.reshape(1, -1)])  # Update policies tensor
            self.rewards = torch.cat([self.rewards, self.reward_list.reshape(1, -1)])  # Update rewards tensor
            self.steps = torch.cat([self.steps, tensor([self.step])])  # Update steps tensor

            self.experiences = []  # Reset experiences
            self.step = 0  # Reset step counter
            self.policy_list = tensor([np.nan] * CONFIG['max_number_of_steps'])  # Reset policy list
            self.reward_list = tensor([np.nan] * CONFIG['max_number_of_steps'])  # Reset reward list

            self.episode += 1  # Increment episode counter

        step_dict = {"state": distances, "output": output, "reward": reward, "action": action, "one_hot": one_hot}  # Create step dictionary
        self.experiences.append(step_dict)  # Append step to experiences
        self.policy_list[self.step] = step_dict["output"] @ step_dict["one_hot"]  # Update policy list
        self.reward_list[self.step] = step_dict["reward"]  # Update reward list
        self.step += 1  # Increment step counter
        self.step_interval = 0  # Reset step interval

    def process_episode_end(self):
        """Processes the end of an episode, including updating the policies and rewards, and resetting the experience list."""
        self.racecar = end_episode(self.racecar)  # End the episode
        reward = CONFIG['collision_reward'] if p.getContactPoints(self.racecar, self.simpleMap) else CONFIG['timeover_reward']  # Calculate reward
        self.policies = torch.cat([self.policies, self.policy_list.reshape(1, -1)])  # Update policies tensor
        self.rewards = torch.cat([self.rewards, self.reward_list.reshape(1, -1)])  # Update rewards tensor
        self.steps = torch.cat([self.steps, tensor([self.step])])  # Update steps tensor

        self.experiences = []  # Reset experiences
        self.step = 0  # Reset step counter
        self.policy_list = tensor([np.nan] * CONFIG['max_number_of_steps'])  # Reset policy list
        self.reward_list = tensor([np.nan] * CONFIG['max_number_of_steps'])  # Reset reward list

        self.episode += 1  # Increment episode counter

    def process_epoch_end(self):
        """Processes the end of an epoch, including updating the policy, calculating reward increase rate, and saving the results."""
        self.epoch += 1  # Increment epoch counter
        self.average_reward = (self.rewards.nansum(dim=1)).mean()  # Calculate average reward
        self.reward_increase_rate = 0  # Initialize reward increase rate
        if self.epoch == 1:
            print(f'[{self.epoch} epoch] average reward: {self.average_reward:.2f}')  # Print average reward for the first epoch
        else:
            prev_mean_reward = (self.pre_rewards.nansum(dim=1)).mean()  # Calculate previous mean reward
            current_mean_reward = (self.rewards.nansum(dim=1)).mean()  # Calculate current mean reward
            self.reward_increase_rate = ((current_mean_reward - prev_mean_reward) / abs(prev_mean_reward) * 100) if prev_mean_reward != 0 else 0  # Calculate reward increase rate
            print(f'[{self.epoch} epoch] average reward: {self.average_reward:.2f} (increase rate: {self.reward_increase_rate:.2f}%)')  # Print average reward and increase rate

        self.pre_rewards = self.rewards  # Update previous rewards
        self.graph_x.append(self.epoch)  # Update graph X values
        self.graph_y.append(self.rewards.nansum()/10)  # Update graph Y values
        plt.plot(self.graph_x, self.graph_y)  # Plot the graph

        update_policy(self.rewards, self.policies, self.steps, self.optimizer)  # Update the policy
        
        self.record_rewards.append(float(self.rewards.nansum(dim=1).mean()))  # Record rewards
        self.experiences = []  # Reset experiences
        self.rewards = tensor([])  # Reset rewards tensor
        self.policies = tensor([])  # Reset policies tensor
        self.steps = tensor([])  # Reset steps tensor
        self.policy_list = tensor([np.nan] * CONFIG['max_number_of_steps'])  # Reset policy list
        self.reward_list = tensor([np.nan] * CONFIG['max_number_of_steps'])  # Reset reward list
        self.episode = 0  # Reset episode counter

    def run(self):
        """Runs the reinforcement learning process, including stepping through simulations, processing episodes and epochs, and saving results."""
        while True:
            p.stepSimulation()  # Step the simulation
            self.t += 0.01  # Increment time step counter

            set_camera(self.racecar)  # Set the camera position

            self.step_interval += 0.01  # Increment step interval
            if self.step_interval >= 0.1:
                if self.epoch < CONFIG['epochs']:  # Check if the maximum number of epochs is not reached
                    if self.episode < CONFIG['episodes']:  # Check if the maximum number of episodes is not reached
                        self.process_step()  # Process a single step
                    else:
                        self.process_epoch_end()  # Process the end of an epoch

                        if (self.epoch > 1 and self.reward_increase_rate == 0) or self.average_reward > CONFIG['target_reward']:  # Check for stopping conditions
                            break
                else:
                    break

        # Save model
        save_model_path = "model/"  # Define model save path
        save_model(self.model, f"{save_model_path}model_weight.pth")  # Save the model

        # Show graph
        save_graph_path = "output/graph/"  # Define graph save path
        save_graph(self.graph_x, self.graph_y, f"{save_graph_path}epoch-reward.png")  # Save the graph

        # Save numeric data (xlsx format)
        save_excel_path = "output/excel/"  # Define excel save path
        save_data(self.record_rewards, f"{save_excel_path}reward_data.xlsx")  # Save the rewards data to excel

        print('finish.')  # Print finish message