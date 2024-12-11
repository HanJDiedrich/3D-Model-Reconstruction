import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d
from scipy.spatial import Delaunay

# bounding box pruning
def bounding_box_pruning_RGB(pts2L,pts2R,pts3, colorsL, colorsR, boxLimits):
    # pts2L, pts2R,pts3 from reconstruct
    
    num_points = pts3.shape[1]
    indices2keep = []
    xLo, xHi, yLo, yHi, zLo, zHi = boxLimits
    for i in range(num_points):
        x = pts3[0][i]
        y = pts3[1][i]
        z = pts3[2][i]
        if (x>=xLo and x<=xHi and
                y>=yLo and y<=yHi and
                z>= zLo and z<=zHi):
            indices2keep.append(i)

    # Shape is (points, dimensions) after
    # remove points outside the bounding box
    pts2L_boxed = pts2L.T[indices2keep]
    pts2R_boxed = pts2R.T[indices2keep]
    pts3_boxed = pts3.T[indices2keep]
    colorsL_boxed = colorsL[indices2keep]
    colorsR_boxed = colorsR[indices2keep]

    return pts2L_boxed.T, pts2R_boxed.T, pts3_boxed.T, colorsL_boxed, colorsR_boxed

# Triangle pruning
def triangle_pruning_RGB(pts2L,pts2R,pts3, colorsL, colorsR, trithresh):

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
    colorsR_triangled = colorsL[valid_points]
    colorsL_triangled = colorsR[valid_points]

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


    return pts2L, pts2R, pts3, colorsL_triangled, colorsR_triangled, remapped_triangles


'''
Based on Professor Charless Fowlkes meshutils.writeply()
'''
def save_point_cloud_RGB(pts3, colors, tri, filename, folder='point_clouds_RGB'):
    """
    Save out a triangulated mesh to a ply file
    
    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        vertex coordinates shape (3,Nvert)
        
    color : 2D numpy.array (dtype=float)
        vertex colors shape (3,Nvert)
        should be float in range (0..1)
        
    tri : 2D numpy.array (dtype=float)
        triangular faces shape (Ntri,3)
        
    filename : string
        filename to save to    
    """
    # Create folder and file path
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, filename)

    #convert colors to proper format
    colors = (colors * 255).astype(np.uint8)

    # Save PLY file
    with open(full_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts3.shape[1]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {tri.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Write point data with colors
        for i in range(pts3.shape[1]):
            f.write('%f %f %f %i %i %i\n' % (pts3[0,i],pts3[1,i],pts3[2,i],colors[0,i],colors[1,i],colors[2,i]))
        
        # Write face data (Triangles)
        for t in range(tri.shape[0]):
            f.write('3 %d %d %d\n' % (tri[t,1],tri[t,0],tri[t,2]))

    print(f".PLY saved to: {full_path}")



def writeply(X,color,tri,filename):
    """
    Save out a triangulated mesh to a ply file
    
    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        vertex coordinates shape (3,Nvert)
        
    color : 2D numpy.array (dtype=float)
        vertex colors shape (3,Nvert)
        should be float in range (0..1)
        
    tri : 2D numpy.array (dtype=float)
        triangular faces shape (Ntri,3)
        
    filename : string
        filename to save to    
    """
    f = open(filename,"w")
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex %i\n' % X.shape[1])
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    f.write('element face %d\n' % tri.shape[0])
    f.write('property list uchar int vertex_indices\n')
    f.write('end_header\n')

    C = (255*color).astype('uint8')
    
    for i in range(X.shape[1]):
        f.write('%f %f %f %i %i %i\n' % (X[0,i],X[1,i],X[2,i],C[0,i],C[1,i],C[2,i]))
    
    for t in range(tri.shape[0]):
        f.write('3 %d %d %d\n' % (tri[t,1],tri[t,0],tri[t,2]))

    f.close()