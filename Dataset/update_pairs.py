#!/usr/bin/env python
# coding: utf-8



# This script is to generate a new csv helper file containing the pairs' info.

import csv
import os

# Define the path to the sample pairs CSV file, adjust it if needed
game_name = 'fps'
csv_file_path = f"{game_name}_pairs.csv"

# Define the path to save the new sample pairs CSV file, adjust it if needed
new_csv_file_path = f"new_{game_name}.csv"

# Define the path to the frames directory, adjust it if needed
image_dir_path = f"./{game_name}/frames/"

# Read the CSV file
with open(csv_file_path, "r") as csv_file:
    reader = csv.reader(csv_file)
    data = list(reader)

# Skip the first line (header)
data = data[1:]

# Initialize start and end numbers, as in each gameplay session, the game begins after 3 seconds.(6 frames per second)
# The first frame will begin at 21st frame (uniformly extract 32 frome 36 frames in each 6-second pair)
start_num = 21
end_num = 52


# Initialize previous base name
prev_base_name = ""

# List to store updated rows
updated_data = []

# Iterate over each row in the CSV data
for row in data:
    # Get the base name from the first column (without extension)
    base_name = os.path.splitext(os.path.basename(row[0]))[0]
    base_name = base_name + '/'

    # If the base name has changed, reset the start and end numbers
    if base_name != prev_base_name:
        start_num = 21
        end_num = 52
        prev_base_name = base_name

    # Define the start and end image paths
    start_image_path = os.path.join(image_dir_path, base_name, f"{start_num:04d}.png")
    end_image_path = os.path.join(image_dir_path, base_name, f"{end_num:04d}.png")


    # Check if the start and end images exist
    if os.path.exists(start_image_path) and os.path.exists(end_image_path):
        # Add the start and end image paths to the row
        row.extend([start_image_path, end_image_path])
        print(f"Images found and added for base name {base_name}")
    else:
        print(f"Images not found for base name {base_name}")

    # Add the updated row to the list
    updated_data.append(row)

    # Update start and end numbers for the next row
    start_num += 6
    end_num += 6

# Write the new data back to the CSV file
with open(new_csv_file_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    # Write the column names
    writer.writerow(['video_path', 'start_time', 'end_time', 'arousal_change','start_frame','end_frame'])
    # Write data
    writer.writerows(data)



# In[23]:


# This part is to remove the rows that is empty in 'start_frame'
import pandas as pd
df = pd.read_csv (new_csv_file_path)


# In[24]:


# remove the rows that is empty in 'start_frame'

df = df[df['start_frame'].notna()]


# In[25]:


up = 0
down = 0
same = 0
total = 0
for index, row in df.iterrows():
    total = total + 1
    if (row['arousal_change'] == 'up'):
        up = up + 1
    elif (row['arousal_change'] == 'down'):
        down = down + 1
    elif (row['arousal_change'] == 'same'):
        same = same + 1
print(total,up,down,same)


# In[26]:


# update the helper file

df.to_csv(new_csv_file_path, index=False)

