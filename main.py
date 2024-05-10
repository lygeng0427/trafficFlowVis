import os
import subprocess
import traceback
import yaml
from pathlib import Path
import logging
import time
import numpy as np
import pandas as pd
import hydra
import pickle
import json
from PIL import Image
from moviepy.editor import VideoFileClip
import h5py
from joblib import dump
from omegaconf import OmegaConf
import torch
from scipy.optimize import linear_sum_assignment

from utils import *

def extract_frames(cfg):
    # Ensure output directory exists
    if not os.path.exists(cfg.clip_frames_path):
        os.makedirs(cfg.clip_frames_path)

    # Load the video file
    video = VideoFileClip(cfg.video_path)

    # Clip the video
    subclip = video.subclip(cfg.start_time, cfg.end_time)

    # Iterate over each frame in the subclip
    for i, frame in enumerate(subclip.iter_frames()):
        
        # Convert raw NumPy array frame to an image
        frame_image = Image.fromarray(frame)
        
        # Save frame to disk
        frame_filename = os.path.join(cfg.clip_frames_path, f"frame_{i:04d}.png")
        frame_image.save(frame_filename)
    print("Frames extracted successfully.")


def extract_frames_per_second(video_path, output_dir='frames'):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video file
    clip = VideoFileClip(video_path)
    
    # Duration of the video in seconds
    duration = int(clip.duration)
    
    # Extract one frame per second
    for i in range(0, duration):
        frame = clip.get_frame(i)

        frame_filename = os.path.join(output_dir, f"frame_{i:04d}.png")
        clip.save_frame(frame_filename, t=i)

    print("Frames per seconds extracted successfully.")


def count_objects(cfg, image, file):
    frame_num = file.split('_')[-1].split('.')[0]
    frame_data = {'frame': frame_num}
    for obj in cfg.objects:
        mask = (
            get_masks(image, obj, visualize=False)[1]
            .cpu()
            .detach()
            .numpy()
        ) # (num_objects, 4)
        if mask.size == 0:
            frame_data[obj] = 0
            continue
        frame_data[obj] = mask.shape[0]
    return frame_data

# ============================================ #
# ====== get sequential bbox matched ========= #
# ============================================ #

# ========= credit to ChatGPT ================ #
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    x1_tl, y1_tl, x1_br, y1_br = box1
    x2_tl, y2_tl, x2_br, y2_br = box2

    # Intersection area
    x_intersection = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))
    y_intersection = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl))
    intersection_area = x_intersection * y_intersection

    # Union area
    box1_area = (x1_br - x1_tl) * (y1_br - y1_tl)
    box2_area = (x2_br - x2_tl) * (y2_br - y2_tl)
    union_area = box1_area + box2_area - intersection_area

    # IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

def matching_objects(cfg, logger):
    files = os.listdir(cfg.clip_frames_path)

    # === sort the file in name order === #
    sorted_files = sorted(files, key=extract_number)
    tracked_objects = {}
    # ======================================= #
    # Here we loop over every sorted_files and#
    # create item id for them.
    # ======================================= #

    for i,frame in enumerate(sorted_files):
        print(f"current photo {i}")
        image = Image.open(os.path.join(cfg.frames_path, frame))
        for category in cfg.objects:
            print(category)
            mask = (
                get_masks(image, category, visualize=False)[1]
                .cpu()
                .detach()
                .numpy()
            ) # (num_objects, 4)
            if mask.size == 0:
                continue
            for j, mask in enumerate(mask):
                print(f"current mask {j} of {category}")
                bbox = mask.tolist()
                print(bbox)
                
                if tracked_objects:
                    matched = False
                    for obj_id, obj_data in tracked_objects.items():
                        prev_bbox = obj_data["bboxes"][-1]
                        iou = calculate_iou(prev_bbox, bbox)
                        if iou > 0.3:  
                            tracked_objects[obj_id]["bboxes"].append(bbox)
                            matched = True
                            break
                    if not matched:
                        tracked_objects[len(tracked_objects) + 1] = {
                            "category": category,
                            "bboxes": [bbox],
                            "start_frame": i,
                            "end_frame": i
                        }
                else:
                    tracked_objects[1] = {
                        "category": category,
                        "bboxes": [bbox],
                        "start_frame": i,
                        "end_frame": i
                    }
        with open("/scratch/lg3490/tfv/sequential_items.json", "w") as file_writer:
            json.dump(tracked_objects, file_writer)

    for obj_id, obj_data in tracked_objects.items():
        obj_data["end_frame"] += len(obj_data["bboxes"]) - 1
    print("All objects tracked successfully.")
    logger.info("All objects tracked successfully.")

    with open("/scratch/lg3490/tfv/sequential_items.json", "w") as file_writer:
        json.dump(tracked_objects, file_writer)
    return tracked_objects
        


def frames_count_objects(cfg, logger):
    print("GPU or not?", torch.cuda.is_available())
    columns = ['frame'] + cfg.objects
    df = pd.DataFrame(columns=columns)
    count = 0
    files = os.listdir(cfg.frames_path)
    sorted_files = sorted(files, key=extract_number)
    for file in sorted_files:
        if file.endswith('.png'):
            start_time = time.time()
            image = Image.open(os.path.join(cfg.frames_path, file))
            frame_data = count_objects(cfg, image, file)
            count_objects_duration = time.time() - start_time
            logger.info(f"Time taken for count_objects call: {count_objects_duration} seconds")
            new_row = pd.DataFrame(frame_data, index=[count])
            df = pd.concat([df, new_row], ignore_index=True)
            count += 1
    df.iloc[:,1:] = df.iloc[:,1:].apply(lambda x: (x/x.sum()), axis=1)
    df.to_csv(cfg.csv_output_path, index=False)


def segment_objects(cfg, image):
    for obj in cfg.objects:
        mask = (
            get_masks(image, obj, visualize=False)[0]
            .cpu()
            .detach()
            .numpy()
        ) # (num_objects, H, W)
        if mask.sum() == 0:
            continue
        masked_image = image.copy()
        for i in range(mask.shape[0]):
            breakpoint()
            sep_mask = mask[i].astype('uint8')
            mask_image = Image.fromarray(sep_mask * 255)
            if image.mode == "RGB":
                mask_image = mask_image.convert("L")
            masked_image = Image.composite(
                image, Image.new("RGB", image.size), mask_image
            )
            masked_image.save("/scratch/lg3490/tfv/new_image.png")

def frames_segment_objects(cfg, logger):
    for file in os.listdir(cfg.frames_path):
        if file.endswith('.png'):
            start_time = time.time()
            image = Image.open(os.path.join(cfg.frames_path, file))
            if file != 'frame_0005.png':
                continue
            segment_objects(cfg, image)
            segment_objects_duration = time.time() - start_time
            logger.info(f"Time taken for segment_objects call: {segment_objects_duration} seconds")

# @hydra.main(version_base="1.2", config_path=".", config_name="main.yaml")
# def main(cfg):
#     logging.basicConfig(
#         level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
#     )
#     logger = logging.getLogger()
#     # extract_frames_per_second(cfg.video_path, output_dir=cfg.frames_path)
#     # frames_segment_objects(cfg, logger)
#     frames_count_objects(cfg, logger)
    # extract_frames(cfg)

# ==== get spatial bbox data ==== #
@hydra.main(version_base="1.2", config_path=".", config_name="bbox_spatial.yaml")
def get(cfg):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    matching_objects(cfg, logger)

if __name__ == "__main__":
    print("started")
    get()