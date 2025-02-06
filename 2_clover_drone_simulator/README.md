# Drone Control with ROS

This program is designed for controlling a drone using ROS (Robot Operating System). It provides a command-line interface for performing basic operations such as takeoff, landing, and flights to specified coordinates.

## Features

- **Takeoff**: The drone ascends to a height of 3 meters.
- **Landing**: The drone lands safely.
- **Flight to Local Coordinates**: The drone navigates to specified local coordinates (X, Y).
- **Flight to Global Coordinates**: The drone navigates to specified global coordinates (latitude, longitude).
- **Flight Report**: A log of all events is maintained and saved in JSON format after the flight ends.

## Installation

1. Ensure that ROS and all necessary packages are installed on your computer.
2. Clone the repository:
   ```bash
   git clone <repository-URL>
   cd <repository-name>
