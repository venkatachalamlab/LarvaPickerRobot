"""
This file defines the file paths for the calibration files
and many of the useful parameters that are used for the LarvaPicker
"""


import numpy as np
import datetime
import cv2

from pathlib import Path


date = str(datetime.date.today())
config_dir = Path(__file__).parent
data_dir = Path(__file__).parent.parent / 'data'
dataset_path = data_dir / date

# Calibration files
h_cam_file = config_dir / 'h_cam_coordinates.npy'
h_cam_file_temp = config_dir / 'h_cam_coordinates_temp.npy'
h_rbt_file = config_dir / 'h_rbt_coordinates.npy'
h_file = config_dir / 'h.npy'
z_file = config_dir / 'z.npy'

# Image files, mostly for diagnosing any issues and communicating information between camera and robot
img_file = (config_dir / 'img.jpg').as_posix()
h_img_file = (config_dir / 'h_img.jpg').as_posix()
m_img_file = (config_dir / 'm_img.jpg').as_posix()
bg_file = (dataset_path / 'bg.jpg').as_posix()

dataset_file = dataset_path / 'data.h5'
log_file = dataset_path / 'log.json'
movie = dataset_path / 'movie.mp4'

# image processing behavior
stage_size = 250
margin_size = 150
center_size = 300
crop_size = 32
h_out, w_out = 1024, 1024
larva_area_ranges = np.array(
    [[10, 50],
     [20, 200],
     [120, 1000],
     [1e8, 1e8+1]]
)

# camera properties
cam_h, cam_w = 2048, 2048
px_to_mm = 248.0 / 1024  # mm/pixels conversion rate
cam_slope = cam_h / cam_w

# robot properties
max_robot_x = 250   # mm
max_robot_y = 235   # mm
solution_coordinates = np.array([27, 7, -21], dtype=np.float)
dormitory_coordinates = np.array([13.1, 18.8, -19.1], dtype=np.float)

# a list of basic colors in BGR colorspace
colour_list = [(255, 0, 0),
               (0, 255, 0),
               (0, 0, 255),
               (255, 255, 0),
               (255, 0, 255),
               (0, 255, 255)]
