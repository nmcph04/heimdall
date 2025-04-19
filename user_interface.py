#!/usr/bin/env python3
import os
import time
import json
import shutil
from process_audio.train_classifier import train_model
from process_audio.analyze_audio import analyze_audio

UI_WIDTH = 100

def clear_screen():
    os.system('cls' if os.name == 'nt' else '/usr/bin/clear')

def input_validation(min_val, max_val):
    print('')
    valid_input = False
    while not valid_input:
        # Moves cursor up one line in terminal
        print("\033[1A", end='')
    
        # Clears line in terminal
        print("\x1b[2K", end='')
    
        try:
            user_input = int(input("Select an option: "))
            if user_input <= max_val and user_input >= min_val:
                valid_input = True
        except ValueError:
            continue
    
    return user_input

def read_settings():
    if not os.path.exists("settings.json"):
        if not os.path.exists("default_settings.json"):
            raise Exception("default_settings.json is missing!")
        
        else:
            shutil.copyfile('default_settings.json', 'settings.json')
    
    with open('settings.json') as f:
        return json.load(f)

def ui_train(settings: dict):
    clear_screen()
    print(f"{"Heimdall Training" :^{UI_WIDTH}}")
    print('-'*UI_WIDTH)
    # Confirm settings, data paths, etc. Then run the classifier training
    print(f"{"Confirm the following settings:" :^{UI_WIDTH}}")
    
    model_dir = settings['modelPath']
    print(f"{f"Model path: {model_dir}" :^{UI_WIDTH}}")
    
    print(f"{f"Confirm directory deletion: {settings['confirmDirectoryDeletion']}" :^{UI_WIDTH}}")
    
    data_dir = settings['trainingDataPath']
    print(f"{f"Training data path: {data_dir}" :^{UI_WIDTH}}")
    
    user_confirm = input("Confirm these settings [y/N] ")
    
    if user_confirm != 'y':
        print(f"{"Settings not confirmed. Going back..." :^{UI_WIDTH}}")
        time.sleep(1)
        return
    
    print("Training classification model...")
    
    train_model(data_dir=data_dir, return_model=False, save_dir=model_dir)
    
    input("Model training complete! Press enter to continue: ")
    return



def ui_analysis(settings: dict):
    while True:
        print(f"{"Heimdall Analysis" :^{UI_WIDTH}}")
        print('-'*UI_WIDTH)
        print('')

        print(f"{"1) Run Analysis" :^{UI_WIDTH}}")
        print(f"{"2) Return to Main Menu" :^{UI_WIDTH}}")

        print('')
        print('-'*UI_WIDTH)

        user_input = input_validation(1, 2)

        if user_input == 1:
            # Confirm paths of models and audio file. Then run the models on the file
            print(f"{"Confirm the following settings:" :^{UI_WIDTH}}")

            model_path = settings['modelPath']
            print(f"{f"Model path: {model_path}" :^{UI_WIDTH}}")

            # Maybe make this its own directory
            data_path = settings['analysisDataPath']
            print(f"{f"Data path: {data_path}" :^{UI_WIDTH}}")

            user_confirm = input("Confirm these settings [y/N] ")

            if user_confirm != 'y':
                print(f"{"Settings not confirmed. Going back..." :^{UI_WIDTH}}")

            print("Running analysis...")

            analyze_audio(audio_file=data_path, models_path=model_path)

            input("File analysis completed. Press enter to return to main menu: ")
            return

        else:
            return

def ui_settings(settings: dict):
    print("WIP")
    input(f"For now, directly modify settings.json. ")

def ui():
    settings_json = read_settings()
    while True:
        clear_screen()
        # Print title
        print(f"{"Heimdall" :^{UI_WIDTH}}")
        print(f"{"Audio Keylogger" :^{UI_WIDTH}}")
        print('-'*UI_WIDTH)
        print('')

        # Print options
        print(f"{"1) Train model " :^{UI_WIDTH}}")
        print(f"{"2) Run analysis" :^{UI_WIDTH}}")
        print(f"{"3) Settings    " :^{UI_WIDTH}}")
        print(f"{"4) Exit        " :^{UI_WIDTH}}")

        print('')
        print('-'*UI_WIDTH)

        user_input = input_validation(min_val=1, max_val=4)
            
        if user_input == 1:
            ui_train(settings=settings_json)
        elif user_input == 2:
            ui_analysis(settings=settings_json)
        elif user_input == 3:
            ui_settings(settings=settings_json)
        else:
            break



def main():
    # Exits gracefully when CTRL+C'd
    try:
        ui()
    except KeyboardInterrupt:
        print('')

if __name__ == '__main__':
    main()