#!/usr/bin/env python
# coding: utf-8
"""
This script is to generate a helper file which help to transform the game data into sample pairs
"""

import pandas as pd
import csv
import os
import glob
import argparse 

def main():
    parser = argparse.ArgumentParser(description='Generate a helper file to transform game data into sample pairs.')
    parser.add_argument('--game_name', type=str, required=True, help='Name of the game (e.g., "fps").')
    parser.add_argument('--data_file', type=str, default='clean_data.csv', help='Path to the clean data CSV file (default: clean_data.csv).')
    args = parser.parse_args()

    game_name = args.game_name
    data_file = args.data_file
    
    # Set the folder path dynamically based on game_name
    folder_path = f'./{game_name}/videos'
    print(f"Using video folder path: {folder_path}")

    # Create the output CSV file name based on game_name
    game_file_name = f'{game_name}_pairs.csv'
    
    # create a csv file with columns name as video_path,start,end,label.
    # This is to calculate the arousal value, generate the label for each time windows, and store the info about each sample pairs. 
    # This file will help prepare the dataset in later parts
    with open(game_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the column names, 'start' is the start time, 'end' is the end time
        writer.writerow(['video_path', 'start', 'end', 'label'])

    # Check if the video folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist. Please check the game name and folder structure.")
        return

    # Get the names of the video files from the dynamic folder path
    files = glob.glob(os.path.join(folder_path, '*'))
    file_names = [os.path.basename(file) for file in files]
    
    # Use the provided data file path
    try:
        gameplay_info = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: The file '{data_file}' was not found. Please check the path.")
        return

    # This is to generate entries about 3-second time windows in Solid from the clean_data.csv
    down = 0
    up = 0
    total = 0
    
    for file in file_names:
      file_path = os.path.join(folder_path, file)
      print("processing:", file)
      file_name = os.path.splitext(file)[0]
      
      # Split file name based on the dynamic game_name
      id_list = file_name.split(f'_{game_name}_')

      if len(id_list) != 2:
          print(f"Warning: File name '{file_name}' does not match the expected format 'player_id_{game_name}_session_id'. Skipping.")
          continue

      # Find the entries with the correct game sessions
      df1 = gameplay_info.loc[(gameplay_info['[control]player_id']==id_list[0])&(gameplay_info['[control]session_id']==id_list[1])]

      # Four data per 1 second, interval step is to 4, i represents data index, we want to get the first index in each second
      for i in range(0,len(df1),4):
        # If there is no overlap, the first data index is i at the beginning of every 6 seconds(one sample pair) and the last data index is i+23;
        # i+24 is then the index of the first data of the next 6 seconds. 
        if ((i+23)<len(df1)):
          # Here, To calculate start time and end time of each time window
          start = float(i) / 4.0
          end = float(i+24) / 4.0

          # Calculate the arousal value of the corresponding time window based on the time and index.
          df2 = df1.iloc[i:i+24]
          arousal_1 = 0
          arousal_2 = 0

          # The index range for the first 3 seconds is 0-11, including 11, for a total of 12 entries of data
          for j in range(0,12):
            arousal = df2.iloc[j]['[output]arousal']
            arousal_1 = arousal_1+arousal
          # average
          arousal_1 = arousal_1/12
          
          #The index range for the last 3 seconds is 12-23
          for k in range(12,24):
            arousal = df2.iloc[k]['[output]arousal']
            arousal_2 = arousal_2+arousal
          arousal_2 = arousal_2/12
          
          # keep labels simple, arousal unchanging-->"same", arousal decreasing-->"down", arousal increasing-->"up"
          arousal_change = "same"
          if(arousal_1>arousal_2):
            arousal_change = "down"
            down = down+1
          if(arousal_1<arousal_2):
            arousal_change = "up"
            up = up+1
          total = total + 1

          with open(game_file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([file_path, start, end, arousal_change])

    # As two pairs cannot extract corresponding frames from videos, then removed in later process
    print("down:", down, "; up:", up, "total:", total)

if __name__ == '__main__':
    main()