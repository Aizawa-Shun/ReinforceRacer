# ReinforceRacer: AI-Powered Autonomous Driving Simulator

**ReinforceRacer: AI-Powered Autonomous Driving Simulator** is an advanced simulation framework designed to train and evaluate reinforcement learning models for autonomous driving. Built on the PyBullet physics engine, this simulator provides a realistic and interactive environment for developing AI algorithms that can control a racecar.

## Features

- **Realistic Simulation**: Utilizes the PyBullet physics engine to provide a detailed and accurate racecar simulation environment.
- **Reinforcement Learning Integration**: Seamlessly integrates reinforcement learning algorithms to control the racecar's steering and velocity.
- **Neural Network Model**: Employs a neural network to decide actions based on LIDAR input and vehicle state.
- **Modular Design**: Structured in a modular fashion, allowing easy updates and customization of various components.
- **Comprehensive Reward System**: Implements a detailed reward mechanism to encourage optimal driving behavior.
- **Data Recording and Analysis**: Includes features to save model weights, plot performance graphs, and export reward data to Excel for further analysis.

## Installation Instructions

To set up and run the **ReinforceRacer: AI-Powered Autonomous Driving Simulator**, follow these steps:

1. **Clone Repository**:

   - Open a terminal (or command prompt) and navigate to the extracted project directory:

   ```sh
   git clone https://github.com/Aizawa-Shun/ReinfoceRacer.git
   ```

2. **Install Required Libraries**:

   - Ensure you have Python installed (preferably version 3.7 or higher). You can download Python from [python.org](https://www.python.org/downloads/).
   - Install the required libraries using pip. Run the following command to install all dependencies listed in the `requirements.txt` file:

   ```sh
   pip install -r requirements.txt
   ```
3. **Run the Simulator**:
   - After installing the necessary libraries, you can run the main simulation script. Execute the following command:
   ```sh
   python main.py
   ```

This will start the ReinforceRacer simulation, and you should see the racecar being controlled by the reinforcement learning agent within the PyBullet environment. If the GUI is enabled, a window will appear showing the racecar in action.

## Change and Adaptation Instructions

1. **Adjust Simulation Settings**:

   - Modify the `CONFIG` dictionary in the `config.py` file to change parameters such as GUI mode, action list, learning rate, and more.

2. **Update the Neural Network**:

   - To change the architecture of the neural network, edit the `NeuralNetwork` class in the `network.py` file.

3. **Modify Reward Functions**:

   - Update the `set_reward` function in the `network.py` file to implement a custom reward system based on different criteria.

4. **Customize the Racecar and Environment**:

   - Edit the URDF files in the `urdf` directory to change the physical properties and appearance of the racecar and the environment.

5. **Extend LIDAR Capabilities**:

   - Adjust the LIDAR setup in the `lidar.py` module to change the sensor configuration and detection parameters.

6. **Implement New Actions**:
   - Add or modify actions in the `CONFIG['ACTION_LIST']` in the `config.py` file to explore different control strategies.

## Contributing

Contributions are welcome! If you have any improvements or new features to add, feel free to fork the repository and create a pull request. Please ensure that your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact [email](tech.code.team@gmail.com).

---

Enjoy using ReinforceRacer and happy coding!
