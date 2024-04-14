import os
import subprocess
import traceback
import yaml
from pathlib import Path
import logging
import time
import numpy as np
import hydra
import pickle
import json
from PIL import Image
from moviepy.editor import VideoFileClip
import h5py
from joblib import dump
from omegaconf import OmegaConf

from utils import *


def extract_frames(video_path, output_dir='frames'):
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


def segment_objects(cfg, image):
    for obj in cfg.objects:
        mask = (
            get_masks(image, obj, visualize=False)[0]
            .cpu()
            .detach()
            .numpy()
        )
        if mask.sum() == 0:
            continue
        else:
            mask = mask[0]

        combined_mask = mask.astype('uint8')
        mask_image = Image.fromarray(combined_mask * 255)
        if image.mode == "RGB":
            mask_image = mask_image.convert("L")
        masked_image = Image.composite(
            image, Image.new("RGB", image.size), mask_image
        )
        
# def video_segment_objects(cfg, logger, video_path):
#     with VideoFileClip(video_path) as video:
#         for frame in video.iter_frames():
#             breakpoint()
#             image = Image.fromarray(frame)
#             segment_objects(cfg, image)

def frames_segment_objects(cfg, logger):
    for file in os.listdir(cfg.frames_path):
        if file.endswith('.png'):
            start_time = time.time()
            image = Image.open(os.path.join(cfg.frames_path, file))
            if file == 'frame_0005.png':
                breakpoint()
            segment_objects(cfg, image)
            segment_objects_duration = time.time() - start_time
            logger.info(f"Time taken for segment_objects call: {segment_objects_duration} seconds")

@hydra.main(version_base="1.2", config_path=".", config_name="main.yaml")
def main(cfg):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()

    # extract_frames(cfg.video_path, output_dir=cfg.frames_path)
    frames_segment_objects(cfg, logger)


if __name__ == "__main__":
    main()