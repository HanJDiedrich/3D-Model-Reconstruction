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


Meshlab
Tutorial followed: https://www.youtube.com/watch?v=4g9Hap4rX0k&list=PLCD0ACE18D723B6C6&index=9

Glue here -> on first mesh

select point based gluing on the next point cloud

manually align range maps to be in the same view position to maximize overlapping area
match coresspondences by clicking on the image, select 4 corresponding points

I selected points on head, left hand, right had, hip, protruding knee, foot, base

all meshes aligned, then i use meshlab to flatten all visible meshes to merge them into one mesh
select surface reconsruction screened poisson
a newwer version of poisson that is better at preserving details and features
select reconstruction depth, recommended to stay under 14, it takes a long time


I started off without exporting colors, then i found a solution and created slightly modified funcitons to export colors along side the meshes
since the object is wooden, the colors came out looking slightly more gray

A large issue i ran into was my bounding box wwas cutting off too many points and i had to adjust them manually checking because the graphs visualized weren't good enough to go off of, so by trial and error i solves an issue in Grab0 and grab 4 where the objects head was completely dissapearing and in grab 3 where the base was disapering and getting cut off

upplaoded all meshes to meshlab with color, had to converyt color type to int from float32 to make sure color values showed up as proper 255 form, not 0-1 form where everything turned out black. Had to be careful with transposing the points to perform these operations.
Used built in mesh lab tools to adjust point size for easier visualization and correspondence and feature and point matching to merge meshes.