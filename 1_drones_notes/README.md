# Drone Notes Program

## Description
This program allows users to save and view notes about drones (UAVs - Unmanned Aerial Vehicles). It is designed as a console application with basic text file operations: adding new notes, viewing saved notes, and exiting the program. Notes are saved in the `notes.txt` file and include a timestamp for easy reference.

## Key Features
- **Add a new note**: Users can enter text for a drone-related note, which is then saved automatically in `notes.txt` with a timestamp.
- **View saved notes**: All saved notes are displayed on the screen. Each note includes a timestamp for easy reference.
- **Exit**: The program can be exited through the main menu.

## Program Structure
The program includes the following functions:
1. **`main_menu()`** — The main menu that allows the user to select one of the available options: add a new note, view saved notes, or exit the program.
2. **`add_note()`** — The function for adding a note. It prompts for note text, adds a timestamp, and saves the note to `notes.txt`.
3. **`view_notes()`** — The function for viewing saved notes. It displays all notes from `notes.txt` on the screen or notifies the user if the file doesn’t exist or there are no notes yet.

## Requirements
- Python 3.x or higher

## Installation and Launch
1. Download or clone the repository containing the program.
2. Open the project in your IDE (e.g., PyCharm).
3. Run the `drone_notes.py` file.

## Usage
Once the program starts, a menu with three options will appear on the screen:

1. **Add a new drone note**:
    - Enter `1` and press Enter.
    - Type the note text (e.g., specifications or usage experience).
    - The note will be saved in `notes.txt` with a timestamp.

2. **View saved drone notes**:
    - Enter `2` and press Enter.
    - All saved notes with timestamps will be displayed on the screen.
    - If the `notes.txt` file doesn’t exist or there are no notes yet, a message will be displayed.

3. **Exit**:
    - Enter `3` and press Enter to exit the program.

## Example Usage
### Adding Notes
```plaintext
Enter drone note text: DJI Mavic Air 2: good camera, stable connection.
Note saved successfully!
