## Introduction
An acoustic keylogger is a type of side-channel attack that takes advantage of a microphone to record and extract what is typed by a target.

## Project Overview
This project aims to create an open-source tool that can extract keystrokes from an audio recording.

### Pipeline:
1. Record audio (and keystrokes for training)
2. Process audio <br>
    a. Separate individual keystrokes<br>
    b. Correlate recorded keystrokes with the keystroke audio (for training)<br>
    c. Clean the audio
3. Train/query model