#!/usr/bin/env python
# coding: utf-8

"""
This script is to extract frames from videos for corresponding sample pairs.
It reads a CSV file with video paths and time windows, and uses ffmpeg to
extract frames from each video.
"""

import pandas as pd
import os
import glob
import subprocess
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from video files based on a sample pair CSV.'
    )
    parser.add_argument(
        '--game_name',
        type=str,
        required=True,
        help='Name of the game (e.g., "solid", "endless").'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=6,
        help='Frames per second to extract from the video (default: 6).'
    )
    
    args = parser.parse_args()
    
    game_name = args.game_name
    output_fps = args.fps

    # Get the path to the pairs CSV file, which is created by the prepare_sample_pair.py script
    csv_file = f'{game_name}_pairs.csv'
    
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: The file '{csv_file}' was not found. Please run 'prepare_sample_pair.py --game_name {game_name}' first.")
        return

    # Read the CSV data
    try:
        data = pd.read_csv(csv_file)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_file}' is empty.")
        return

    print(f"Successfully loaded data from {csv_file}")

    # Set the destination folder where the extracted frames will be stored
    dest_dir = f"./{game_name}/frames/"
    
    # Get the path to the folder where the videos are stored, based on the game name
    v_dir = f'./{game_name}/videos/'

    # Iterate through all entries in the DataFrame
    # This loop processes each video path found in the CSV
    for index, row in data.iterrows():
        video_path = row['video_path']
        
        # Process the video path to create a save directory for frames
        # The save path is based on the video's file name (without extension)
        file_name = os.path.basename(video_path)
        base_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(dest_dir, base_name)
        
        # Ensure the save directory ends with a slash for ffmpeg
        if not save_path.endswith(os.sep):
            save_path += os.sep
            
        # Create the output directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Created directory: {save_path}")
        
        # Prepare the ffmpeg command to extract frames
        # -i: input file
        # -vf "fps=...": video filter to set the output frame rate
        # "%04d.png": output file naming format (e.g., 0001.png, 0002.png)
        command = f'ffmpeg -i "{video_path}" -vf "fps={output_fps}" "{save_path}%04d.png"'
        
        print(f"\nProcessing video: {video_path}")
        print(f"Saving frames to: {save_path}")
        print("--------------------------")
        
        # Execute the ffmpeg command
        # `shell=True` is used for simplicity, but be careful with untrusted input
        try:
            subprocess.run(command, shell=True, check=True)
            print("Extraction finished.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing ffmpeg command for {video_path}: {e}")
        except FileNotFoundError:
            print("Error: 'ffmpeg' command not found. Please ensure ffmpeg is installed and added to your system's PATH.")
            break # Exit the loop if ffmpeg is not found

    print("\nFrame extraction process completed.")

if __name__ == "__main__":
    main()