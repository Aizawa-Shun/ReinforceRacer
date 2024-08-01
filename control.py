import pybullet as p
import random

def control_steering(racecar, angle, steeringLinks):
    """Controls the steering of the racecar.
    
    Args:
        racecar (int): The unique ID of the racecar in the simulation.
        angle (float): The target steering angle.
        steeringLinks (list): List of joint indices for the steering mechanism.
    """
    for steer in steeringLinks:
        p.setJointMotorControl2(racecar, steer, controlMode=p.POSITION_CONTROL, targetPosition=angle)  # Set the steering angle

def control_velocity(racecar, vel, wheelLinks):
    """Controls the velocity of the racecar.
    
    Args:
        racecar (int): The unique ID of the racecar in the simulation.
        vel (float): The target velocity.
        wheelLinks (list): List of joint indices for the wheels.
    """
    for wheel in wheelLinks:
        p.setJointMotorControl2(racecar, wheel, controlMode=p.VELOCITY_CONTROL, targetVelocity=vel)  # Set the wheel velocity

def set_camera(racecar):
    """Sets the camera to follow the racecar.
    
    Args:
        racecar (int): The unique ID of the racecar in the simulation.
    """
    focus_position, _ = p.getBasePositionAndOrientation(racecar)  # Get the position of the racecar
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-90, cameraPitch=-70, cameraTargetPosition=focus_position)  # Set the camera position

def end_episode(agent):
    """Ends the current episode and resets the racecar.
    
    Args:
        agent (int): The unique ID of the racecar in the simulation.
    
    Returns:
        int: The unique ID of the new racecar.
    """
    p.removeBody(agent)  # Remove the racecar from the simulation
    iniPosture = [0, 0, random.uniform(-1.047, 1.047)]  # Generate a random initial posture
    iniPosture = p.getQuaternionFromEuler(iniPosture)  # Convert from Euler angles to Quaternion
    agent = p.loadURDF("racecar/racecar.urdf", baseOrientation=iniPosture, flags=p.URDF_USE_SELF_COLLISION)  # Load a new racecar with self-collision enabled
    return agent  # Return the new racecar ID
