import time
from pynput import keyboard
import threading
import tkinter as tk
import os
import pyaudio
import wave

TEXT_FILENAME = "keylog.tsv"
# file format: 
#   pressed: 1 for press, 0 for release
#   timestamp: timestamp for event (ms)
#   key: key associated with event

# Deletes keylog.txt if it exists
if os.path.isfile(TEXT_FILENAME):
    os.remove(TEXT_FILENAME)

# Global variables for controlling the recording process
recording = False
running = True

def on_press(key):
    global chars, time_since_start
    print(chars, end='\r')
    chars += 1
    with open(TEXT_FILENAME, 'a') as f:
        timestamp = int((time.time() - time_since_start)*1000) # milliseconds since the script was started (about)
        f.write(f'1\t{timestamp}\t{key}\n')

def on_release(key):
    global chars, time_since_start
    print(chars, end='\r')
    chars += 1
    with open(TEXT_FILENAME, 'a') as f:
        timestamp = int((time.time() - time_since_start)*1000) # milliseconds since the script was started (about)
        f.write(f'0\t{timestamp}\t{key}\n')

def record_audio():
    global recording, running

    if not recording:
        return
    
    FORMAT = pyaudio.paInt16  # Audio format
    CHANNELS = 1              # Number of audio channels
    RATE = 44100              # Sample rate
    CHUNK = 1024              # Buffer size
    WAVE_OUTPUT_FILENAME = "keylog.wav"  # Output file name

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    try:
        while running and recording:
            data = stream.read(CHUNK)
            frames.append(data)
    # Stop recording on end() function is called
    except KeyboardInterrupt: 
        pass

    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recorded data as a WAV file
        with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

def end():
    global running, recording

    # Stop keylogging
    running = False
    
    # Stop audio recording
    recording = False
    
    # Wait for the threads to finish
    keylogging_thread.join()
    if not threading.main_thread().is_alive():  # Check if main thread is alive
        return
    record_audio_thread.join()

    root.destroy()

chars = 0

# Create a thread for keylogging to run in background
def start_keylogging():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while running:
            time.sleep(0.1)
        listener.stop()

# Set up the main window and stop button
root = tk.Tk()
root.title("Key Logger")

stop_button = tk.Button(root, text="Stop Logging", command=end)
stop_button.pack(pady=20)

time_since_start = time.time()

# Start keylogging in a separate thread
keylogging_thread = threading.Thread(target=start_keylogging, daemon=True)
keylogging_thread.start()

# Start audio recording in a separate thread
recording = True
record_audio_thread = threading.Thread(target=record_audio, daemon=True)
record_audio_thread.start()

# Run the tkinter main loop
root.mainloop()
