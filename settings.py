import pybullet as p
import pybullet_data
import random
import lidar

def setup_simulation(GUI):
    """Sets up the simulation environment.
    
    Args:
        GUI (bool): Flag to enable or disable GUI.
    
    Returns:
        tuple: A tuple containing the physics client, racecar ID, map ID, steering links, and wheel links.
    """
    if GUI:
        physicsClient = p.connect(p.GUI)  # Connect to PyBullet with GUI
    else:
        physicsClient = p.connect(p.DIRECT)  # Connect to PyBullet without GUI

    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Add path to load racecar and plane
    p.setGravity(0, 0, -9.8)  # Set the gravity

    plane = p.loadURDF("plane.urdf")  # Load the floor
    iniPosture = [0, 0, random.uniform(-1.047, 1.047)]  # Generate a random initial posture
    iniPosture = p.getQuaternionFromEuler(iniPosture)  # Convert from Euler to Quaternion
    racecar = p.loadURDF("racecar/racecar.urdf", baseOrientation=iniPosture, flags=p.URDF_USE_SELF_COLLISION)  # Load racecar with self-collision enabled
    simpleMap = p.loadURDF("urdf/simple_map.urdf", basePosition=[0.5, 3.4, 0.2], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)  # Load simple map with self-collision enabled

    steeringLinks = [4, 6]  # List of joint indices for steering
    wheelLinks = [2, 3, 5, 7]  # List of joint indices for wheels

    return physicsClient, racecar, simpleMap, steeringLinks, wheelLinks  # Return the setup components

def get_initial_position(racecar):
    """Gets the initial position of the racecar.
    
    Args:
        racecar (int): The unique ID of the racecar in the simulation.
    
    Returns:
        list: The position of the racecar.
    """
    pos, _ = p.getBasePositionAndOrientation(racecar)  # Get the position and orientation of the racecar
    return pos  # Return the position

def setup_lidar(racecar):
    """Sets up the lidar for the racecar.
    
    Args:
        racecar (int): The unique ID of the racecar in the simulation.
    
    Returns:
        int: The joint index for the lidar.
    """
    hokuyo_joint = 8  # Joint index for the lidar
    lidar.set(racecar, hokuyo_joint)  # Setup lidar
    return hokuyo_joint  # Return the lidar joint index