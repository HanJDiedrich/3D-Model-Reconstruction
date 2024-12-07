import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d
from scipy.spatial import Delaunay

# bounding box pruning
def bounding_box_pruning(pts2L,pts2R,pts3, boxLimits):
    # pts2L, pts2R,pts3 from reconstruct
    
    num_points = pts3.shape[1]
    indices2prune = []
    xLo, xHi, yLo, yHi, zLo, zHi = boxLimits
    for i in range(num_points):
        x = pts3[0][i]
        y = pts3[1][i]
        z = pts3[2][i]
        if not (x>=xLo and x<=xHi and\
            y>=yLo and y<=yHi and\
            z>= zLo and z<=zHi):
            indices2prune.append(i)

    # np.delete returns a new array
    # Shape is (points, dimensions) after
    pts2L_boxed = np.delete(pts2L.T, indices2prune, axis=0)
    pts2R_boxed = np.delete(pts2R.T, indices2prune, axis=0)
    pts3_boxed = np.delete(pts3.T, indices2prune, axis=0)

    return pts2L_boxed, pts2R_boxed, pts3_boxed

# Triangle pruning
def triangle_pruning(pts2L,pts2R,pts3, trithresh):

    triangulateR = Delaunay(pts2R)
    #
    # triangle pruning
    #
    triangles = triangulateR.simplices # indexes

    triangles2prune = [] # hold the complete tri[x,y,z indices]

    for tri_index, tri in enumerate(triangles):
        triX, triY, triZ = pts3[tri[0]], pts3[tri[1]], pts3[tri[2]]
        edges = [np.linalg.norm(triX - triY),
                np.linalg.norm(triY - triZ), 
                np.linalg.norm(triZ - triX)]
        if(max(edges) > trithresh):
            triangles2prune.append(tri_index)

    # already in simplicies form
    valid_triangles = np.delete(triangles, triangles2prune, axis=0)
    valid_points = np.unique(valid_triangles)

    #
    # remove any points which are not refenced in any triangle
    #
    pts2L_triangled = pts2L[valid_points]
    pts2R_triangled = pts2R[valid_points]
    pts3_triangled = pts3[valid_points]

    # transpose back to normal for graphing
    pts2R = pts2R_triangled.T
    pts2L = pts2L_triangled.T
    pts3 = pts3_triangled.T

    return pts2L, pts2R, pts3

# Same as triangle purning but return the triangle meshes
def triangle_pruning_return(pts2L,pts2R,pts3, trithresh):

    triangulateR = Delaunay(pts2R)
    #
    # triangle pruning
    #
    triangles = triangulateR.simplices # indexes

    triangles2prune = [] # hold the complete tri[x,y,z indices]

    for tri_index, tri in enumerate(triangles):
        triX, triY, triZ = pts3[tri[0]], pts3[tri[1]], pts3[tri[2]]
        edges = [np.linalg.norm(triX - triY),
                np.linalg.norm(triY - triZ), 
                np.linalg.norm(triZ - triX)]
        if(max(edges) > trithresh):
            triangles2prune.append(tri_index)

    # already in simplicies form
    valid_triangles = np.delete(triangles, triangles2prune, axis=0)
    valid_points = np.unique(valid_triangles)

    #
    # remove any points which are not refenced in any triangle
    #
    pts2L_triangled = pts2L[valid_points]
    pts2R_triangled = pts2R[valid_points]
    pts3_triangled = pts3[valid_points]

    # transpose back to normal for graphing
    pts2R = pts2R_triangled.T
    pts2L = pts2L_triangled.T
    pts3 = pts3_triangled.T

    # Now we need to remap the triangle indices to match the pruned points
    # valid_triangles are still referencing the original points, so we need to map them to the new indices
    remapped_triangles = []
    for tri in valid_triangles:
        remapped_tri = np.array([np.where(valid_points == idx)[0][0] for idx in tri])
        remapped_triangles.append(remapped_tri)

    # Convert the remapped triangles to a numpy array
    remapped_triangles = np.array(remapped_triangles)


    return pts2L, pts2R, pts3, remapped_triangles



def save_point_cloud(pts3, filename, folder='point_clouds'):
    os.makedirs(folder, exist_ok=True)
    
    # Create full file path
    full_path = os.path.join(folder, filename)
    
    # Create a structured array with x, y, z coordinates
    point_cloud = np.column_stack((pts3[0], pts3[1], pts3[2]))
    
    # Save PLY file
    with open(full_path, 'w') as f:
        # Correct PLY header format
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(point_cloud)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        # Write point data
        np.savetxt(f, point_cloud, fmt='%.6f %.6f %.6f')
    
    print(f"Point cloud saved to: {full_path}")