# UAV Route Calculation Application

A simple Python GUI application for calculating a UAV route based on the starting, ending, and intermediate points. This app allows users to input coordinates, add intermediate waypoints with descriptions, edit or delete them, and calculate the final route.

## Features

- **Enter starting and ending points** with latitude and longitude coordinates.
- **Add intermediate waypoints** with latitude, longitude, and a description.
- Ability to **edit and delete intermediate waypoints**.
- **Calculate route** to display the full path, from the starting point through intermediate waypoints to the ending point.
- **Display calculated route** on the screen.

## Usage

1. Enter the coordinates for the starting and ending points.
2. Add one or more intermediate waypoints by entering their latitude, longitude, and description.
3. You can edit or delete added waypoints.
4. Click the "Calculate Route" button to display the complete route.
5. Use the "Exit" button to close the application.

## Installation

1. Clone the repository or copy the project files.
2. Ensure you have Python 3.x installed.
3. Run the application:
    ```bash
    python drone_route_calculator.py
    ```

## Requirements

- **Python 3.x**
- **tkinter** (usually included with the standard Python library)

## Project Structure

- `drone_route_calculator.py` - main application file.
- `README.md` - project information.

## Notes

- Basic error handling for input validation is included: the application will alert you if any input is incorrect.
- All data is entered in input fields and displayed in a list for easy viewing.

---

### Author

This application was developed to simplify UAV route calculation using Python and tkinter.
