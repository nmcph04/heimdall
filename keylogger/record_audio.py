def record_audio():
    import pyaudio
    import wave

    # Set parameters for recording
    FORMAT = pyaudio.paInt16  # Audio format
    CHANNELS = 1              # Number of audio channels
    RATE = 44100              # Sample rate
    CHUNK = 1024              # Buffer size
    WAVE_OUTPUT_FILENAME = "output.wav"  # Output file name

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    # Records until Ctrl+C
    except KeyboardInterrupt: 
        print("Finished recording.")

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

if __name__ == '__main__':
    record_audio()