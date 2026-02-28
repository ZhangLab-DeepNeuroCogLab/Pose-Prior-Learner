"""
Step 1: Extract frames from video
"""
import sys

sys.path.append("..")
sys.path.append("../..")

import os
import cv2
from tqdm import tqdm

DATA_DIR = "/home/ziyu/Datasets/h36m/"


def get_video_paths(subject):
    video_paths = os.listdir(DATA_DIR + subject + "/Videos/")
    return video_paths


#subjects = ["S1", "S11", "S5", "S6", "S7", "S8", "S9"]
subjects = ["S1", "S7", "S9"]
valid_activities = ['Directions', 'Posing', 'Walking']
# for each subject
for subject in subjects:
    print("Processing " + subject + " ...")
    video_paths = get_video_paths(subject)
    path_prefix = DATA_DIR + subject

    save_prefix = DATA_DIR + subject + "/Frames/"

    for j in tqdm(range(len(video_paths))):
        video_path = path_prefix + "/Videos/" + video_paths[j]
        activity = video_paths[j][:-4]
        if activity.split(' ')[0] in valid_activities:
            save_prefix_vid = save_prefix + activity + "/"
            if not os.path.exists(save_prefix_vid):
                os.makedirs(save_prefix_vid)
            vidcap = cv2.VideoCapture(video_path)
            i = 0
            while True:
                success, image = vidcap.read()
                if not success:
                    break
                cv2.imwrite(save_prefix_vid + ("frame%04d.png" % i), image)  # save frame as JPEG file
                i += 1