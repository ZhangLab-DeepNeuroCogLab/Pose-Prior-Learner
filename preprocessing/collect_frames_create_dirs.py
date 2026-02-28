"""
Step 3: Create directories for collecting frames
"""

import sys
sys.path.append("..")
sys.path.append("../..")

DATA_DIR = "/home/ziyu/Datasets/h36m/"
dst = DATA_DIR + "preprocessed/training"
subjects = ["S1", "S11", "S5", "S6", "S7", "S8", "S9"]
meta = "/home/ziyu/Pose/preprocessing/data_dict.json"

import os
import json_tricks as json
import numpy as np
from tqdm import tqdm

print("Starting to load data_dict.json ..")
with open(meta, 'r') as file:
    center_dict = json.load(file)
print("Finish loading data_dict.json ..")

# vertical scale parameter for affine transformation, check get_affine_transform.py for details
# you might have to play around with this a bit. (effectively it's a zoom factor)
scale_factor = 170

valid_activities = ['Waiting', 'Posing', 'Greeting', 'Directions', 'Discussion', 'Walking', 'Greeting', 'Photo',
                    'Purchases', 'WalkTogether', 'WalkDog']

for subject in subjects:
    print("Processing " + subject + " ...")
    activities = list(center_dict[subject].keys())

    raster_positions = []

    cameras = []

    destination_path = os.path.join(dst, subject)

    for act in tqdm(activities):
        name = ''.join([s for s in act if not s.isdigit()]).replace('.', '')
        if name not in valid_activities:
            continue
        entries = center_dict[subject][act]
        bboxes = [entry['bounding_box'] for entry in entries]
        frames = [entry['frame'] for entry in entries]
        # define bounding boxes
        for bbox, frame in zip(bboxes, frames):
            # divide image into a grid where each bounding box falls into a certain region
            top_left = np.floor(bbox[0] / scale_factor)
            bottom_right = np.ceil(bbox[1] / scale_factor)
            # define folder for every region, ie. all frames in that folder will share the same bounding box and therefore
            # background (across activities)
            raster_pos = ','.join([str(top_left[0]), str(top_left[1]), str(bottom_right[0]), str(bottom_right[1])])
            # find camera id
            cam_id = frame.split('/')[-2].split('.')[1]
            final_path = os.path.join(destination_path, raster_pos + '_' + cam_id)
            if not os.path.exists(final_path):
                os.makedirs(final_path)