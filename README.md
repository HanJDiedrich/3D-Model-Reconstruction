# 3D-Model-Reconstruction
Reconstruct 3D model from 2D images

Utilities->
    calibration.py - Functions for intrinsic and extrinsic camera parameter calibration
    camutils.py - Utility functions for Camera class, and other linear algebra (triangulation, rotation)
    decode.py - Functions for decoding to get gray and color masks, and reconstructing depth map using triangulation
    meshing.py - Functions to perform bounding box pruning and triangle pruning to clean up points and prepare for mesh generation
    visutils.py - Utility functions for visualizing points in 3D and 2D(XY, XZ, YZ)


Steps:

Calibration

Stereo correspondences
- This step was very tricky because it was difficult to find the Dense disparity map and find a way to preserve colors

Triangulation

Clean up using bounding box pruning 
Create mesh using scipy.delauany, clean the mesh up using triangle pruning to remove skinny triangles
and triangle pruning