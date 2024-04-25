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

from utils import *

def extract_frames(cfg):
    # Ensure output directory exists
    if not os.path.exists(cfg.clip_frames_path):
        os.makedirs(cfg.clip_frames_path)

    # Load the video file
    clip = VideoFileClip(cfg.video_path)

    # Clip the video
    subclip = video.subclip(cfg.start_time, cfg.end_time)

    # Iterate over each frame in the subclip
    for i, frame in enumerate(subclip.iter_frames()):
        
        # Convert raw NumPy array frame to an image
        frame_image = Image.fromarray(frame)
        
        # Save frame to disk
        frame_image.save(f"spatial/frames/frame_{i:03d}.png")


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

    print("Frames extracted successfully.")


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

@hydra.main(version_base="1.2", config_path=".", config_name="main.yaml")
def main(cfg):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    # extract_frames_per_second(cfg.video_path, output_dir=cfg.frames_path)
    # frames_segment_objects(cfg, logger)
    frames_count_objects(cfg, logger)


if __name__ == "__main__":
    main()