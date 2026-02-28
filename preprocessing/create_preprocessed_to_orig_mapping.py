"""
Step 5: Create mapping from preprocessed frame to original frame
    - Creates preprocessed_to_orig.json
"""

import sys

sys.path.append("..")
sys.path.append("../..")

import json_tricks as json
import cdflib
from tqdm import tqdm

########################
# CONSTANTS

CODE_DIR = "/home/ziyu/Pose/"
DATA_DIR = "/home/ziyu/Datasets/h36m/"
PREPROCESSED_TO_ORIG = "/home/ziyu/Pose/preprocessing/preprocessed_to_orig.json"
########################

"""
Create preprocessed to original frame, bbox, pose mapping
"""
# get original bounding box and pose information
print("Starting to load data_dict.json ..")
with open(CODE_DIR + "preprocessing/data_dict.json", "r") as f:
    data_dict = json.load(f)
print("Finish loading data_dict.json ..")

# get original bounding box and pose information
print("Starting to load mapping_preprocessed.json..")
with open(CODE_DIR + "preprocessing/mapping_preprocessed.json", "r") as f:
    mapping = json.load(f)
print("Finish loading data mapping_preprocessed.json ..")

preprocessed_to_orig = {}
for s in ['S9', 'S11']:
    print("Processing " + s + " ...")
    subject = data_dict[s]
    for video in tqdm(subject.keys()):
        frames = subject[video]
        # if this video was preprocessed
        if frames[0]['frame'] in mapping:
            poses = cdflib.CDF(DATA_DIR + s + "/Poses/" + video + ".cdf")
            poses = poses['Pose'].squeeze()

            for i in range(len(frames)):
                path = frames[i]['frame']
                # preprocessed frame --> bounding box, pose
                preprocessed_to_orig[mapping[path]] = {'bbox': frames[i]['bounding_box'],
                                                       'pose': poses[i],
                                                       'path': path}
        else:
            continue

# Get mapping from preprocessed frame path to corresponding bbox and pose
with open("./preprocessed_to_orig.json", "w") as f:
    json.dump(preprocessed_to_orig, f)