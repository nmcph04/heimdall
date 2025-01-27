import pandas as pd
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from pydub.silence import detect_nonsilent
import noisereduce as nr
import soundfile as sf

# Load the audio file
audio_file = "Keystrokes_1.wav"  # Replace with your audio file path
data, sample_rate = sf.read(audio_file)

# Reduce noise
reduced_noise = nr.reduce_noise(y=data, sr=sample_rate, prop_decrease=.75)

# Save the cleaned audio to a new file
output_file = "cleaned_output.wav"
sf.write(output_file, reduced_noise, sample_rate)

audio = AudioSegment.from_wav(output_file)


# Create a list to hold the keystroke segments
keystroke_segments = []
threshold = 20

# Define a function to calculate RMS energy
def calculate_rms(segment):
    return np.sqrt(np.mean(np.square(segment.get_array_of_samples())))

# Analyze the audio in small chunks
chunk_size = 200  # Size of each chunk in milliseconds
for start in range(0, len(audio), chunk_size):
    segment = audio[start:start + chunk_size]
    if calculate_rms(segment) > threshold:  # Define a suitable threshold
        keystroke_segments.append(segment)
    else:
        print("Threshold not met")


# Now you can work with the keystroke segments
# For example, you can export each segment to a file
#for i, segment in enumerate(keystroke_segments):
    #print(i)
    #play(segment)

print(f"Extracted {len(keystroke_segments)} keystroke segments.")