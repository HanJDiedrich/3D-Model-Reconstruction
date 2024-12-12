import numpy as np
import matplotlib.pyplot as plt
from . import camutils

# Convert image to grayscale and float to use for decoding gray -> binary
def image_convert(image):
    # If color, convert to grayscale
    if len(image.shape) == 3:
        image = image.mean(axis=-1)
        
    # Check image in integer format
    if image.dtype == np.uint8:
        # Normalize to pixel values
        image = image.astype(float)/255.0
        
    return image


def decode_gray(imprefix,start,threshold):
    """
    Given a sequence of 20 images of a scene showing projected 10 bit gray code, 
    decode the binary sequence into a decimal value in (0,1023) for each pixel.
    Mark those pixels whose code is likely to be incorrect based on the user 
    provided threshold.
    """
    
    # 10 bit code
    nbits = 10
    # Compute mask
    gray_bits = []
        
    # Read and convert image to find mask shape
    image = image_convert(plt.imread(f"{imprefix}{(0):02d}.png"))
    mask = np.ones(image.shape)
    
    # Loop through image pairs (N, N + 1)
    for N in range(start, start + 20, 2): # start, end, increment
        image1 = image_convert(plt.imread(f"{imprefix}{(N):02d}.png"))
        image2 = image_convert(plt.imread(f"{imprefix}{(N+1):02d}.png"))
        
        
        # Compute mask
        gray_bits.append(1 * (image1 > image2))

        mask = mask * (1 * (np.abs(image1 - image2) > threshold))

    # Convert gray code to binary
    binary_code = []
    binary_code.append(gray_bits[0])
    for i in range(1, nbits):
        binary_code.append(binary_code[i-1] ^ gray_bits[i])
                           
    # Convert binary to decimal
    code = np.zeros(image.shape)
        
    # $\sum_{n=0}^9 B[bits - 1 -n]*2^n$
    for i in range(nbits):
        code += binary_code[nbits - 1 - i] * (2**i)

    return code, mask

# Compute color based mask to differentiate objects from background
def decode_color(colorImprefix, threshold):
    image1 = plt.imread(f"{colorImprefix}{(0):02d}.png")
    image2 = plt.imread(f"{colorImprefix}{(1):02d}.png")
    
    # Compute squared differences in RGB space
    color_diff = np.sum((image1 - image2) ** 2, axis=-1)

    # Create mask based on threshold
    color_mask = (color_diff > threshold)

    return color_mask

def reconstruct(imprefixL, imprefixR, gray_threshold, colorImprefixL, colorImprefixR, color_threshold, camL, camR):
    """
    Performing matching and triangulation of points on the surface using structured
    illumination.
    """
    # left
    HL,HmaskL = decode_gray(imprefixL, 0, gray_threshold)
    VL,VmaskL = decode_gray(imprefixL, 20, gray_threshold)
    colorMaskL = decode_color(colorImprefixL, color_threshold)

    # right
    HR,HmaskR = decode_gray(imprefixR, 0, gray_threshold)
    VR,VmaskR = decode_gray(imprefixR, 20, gray_threshold)
    colorMaskR = decode_color(colorImprefixR, color_threshold)

    # Construct the combined 20 bit code C = H + 1024*V and mask for each view
    CL = HL + 1024 * VL
    CR = HR + 1024 * VR
    
    # Account for color mask
    Lmask = HmaskL * VmaskL * colorMaskL
    Rmask = HmaskR * VmaskR * colorMaskR
    
    # Mask undecodeable/invalid
    CR = CR * Rmask
    CL = CL * Lmask

    #intersection, matchR, matchL = np.intersect1d(CR, CL, return_indices=True)
    matchR = np.intersect1d(CR, CL, return_indices=True)[1]
    matchL = np.intersect1d(CL, CR, return_indices=True)[1]

    h,w = CL.shape
    xx,yy = np.meshgrid(range(w),range(h))
    xx = np.reshape(xx,(-1,1))
    yy = np.reshape(yy,(-1,1))
    pts2R = np.concatenate((xx[matchR].T,yy[matchR].T),axis=0)
    pts2L = np.concatenate((xx[matchL].T,yy[matchL].T),axis=0)
    
    pts3 = camutils.triangulate(pts2L,camL,pts2R,camR)

    return pts2L,pts2R,pts3


# Color reconstruction
def reconstruct_with_rgb(imprefixL, imprefixR, gray_threshold, 
                         colorImprefixL, colorImprefixR, color_threshold, 
                         camL, camR):
    """
    Integrate decoding, and RGB extraction 
    to generate a dense 3D point cloud with color information.
    """
    """
    Performing matching and triangulation of points on the surface using structured
    illumination.
    """
    # left
    HL,HmaskL = decode_gray(imprefixL, 0, gray_threshold)
    VL,VmaskL = decode_gray(imprefixL, 20, gray_threshold)
    colorMaskL = decode_color(colorImprefixL, color_threshold)

    # right
    HR,HmaskR = decode_gray(imprefixR, 0, gray_threshold)
    VR,VmaskR = decode_gray(imprefixR, 20, gray_threshold)
    colorMaskR = decode_color(colorImprefixR, color_threshold)

    # Construct the combined 20 bit code C = H + 1024*V and mask for each view
    CL = HL + 1024 * VL
    CR = HR + 1024 * VR
    
    # Account for color mask
    Lmask = HmaskL * VmaskL * colorMaskL
    Rmask = HmaskR * VmaskR * colorMaskR
    
    # Mask undecodeable/invalid
    CR = CR * Rmask
    CL = CL * Lmask

    #intersection, matchR, matchL = np.intersect1d(CR, CL, return_indices=True)
    matchR = np.intersect1d(CR, CL, return_indices=True)[1]
    matchL = np.intersect1d(CL, CR, return_indices=True)[1]

    h,w = CL.shape
    xx,yy = np.meshgrid(range(w),range(h))
    xx = np.reshape(xx,(-1,1))
    yy = np.reshape(yy,(-1,1))
    pts2R = np.concatenate((xx[matchR].T,yy[matchR].T),axis=0)
    pts2L = np.concatenate((xx[matchL].T,yy[matchL].T),axis=0)
    
    pts3 = camutils.triangulate(pts2L,camL,pts2R,camR)
    # Extract colors for matched points
    imageL = plt.imread(f"{colorImprefixL}{(0):02d}.png")
    imageR = plt.imread(f"{colorImprefixR}{(1):02d}.png")

    colorsL = imageL[yy[matchL].flatten(), xx[matchL].flatten()]
    colorsR = imageR[yy[matchR].flatten(), xx[matchR].flatten()]

    return pts2L, pts2R, pts3, colorsL, colorsR
