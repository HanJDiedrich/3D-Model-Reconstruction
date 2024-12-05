import numpy as np
import matplotlib.pyplot as plt

import Utilities.camutils as camutils
import Utilities.calibration as calibration
import Utilities.decode as decode


# Set up chessboard
square_size = 3
chessboardDimensions = (7,9, square_size)
cam1Images = calibration.getImages("/home/hanjdiedrich/UCI-Ubuntu/cs117/calib_jpg_u/frame_C0_*.jpg")
cam2Images = calibration.getImages("/home/hanjdiedrich/UCI-Ubuntu/cs117/calib_jpg_u/frame_C1_*.jpg")

# Intrinsic camera calibration
cam1K = calibration.intrinsic_Calibration(chessboardDimensions, cam1Images, False)
cam2K = calibration.intrinsic_Calibration(chessboardDimensions, cam2Images, False)
print(cam1K)
print(cam2K)

# Extrinsic camera calibration
cam1Image = "/home/hanjdiedrich/UCI-Ubuntu/cs117/calib_jpg_u/frame_C0_01.jpg"
cam2Image = "/home/hanjdiedrich/UCI-Ubuntu/cs117/calib_jpg_u/frame_C1_01.jpg"

cam1, cam1_pts2, cam1_pts3 = calibration.extrinsic_Calibration(chessboardDimensions, cam1Image, cam1K)
print(cam1)
cam2, cam2_pts2, cam2_pts3 = calibration.extrinsic_Calibration(chessboardDimensions, cam2Image, cam2K)
print(cam2)

gray_threshold = 0.01
imagePrefix = "/home/hanjdiedrich/UCI-Ubuntu/cs117/manny/grab_0_u/frame_C0_"
code, mask = decode.decode_gray(imagePrefix, 0, gray_threshold)

color_threshold = 0.005
colorImage1 = "/home/hanjdiedrich/UCI-Ubuntu/cs117/manny/grab_0_u/color_C0_00.png"
colorImage2 = "/home/hanjdiedrich/UCI-Ubuntu/cs117/manny/grab_0_u/color_C0_01.png"
color_mask = decode.decode_color(colorImage1, colorImage2, color_threshold)