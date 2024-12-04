import numpy as np
import matplotlib.pyplot as plt

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
def decode_color(colorImage1, colorImage2, threshold):
    image1 = plt.imread(colorImage1)
    image2 = plt.imread(colorImage2)
    
    # Compute squared differences in RGB space
    color_diff = np.sum((image1 - image2) ** 2, axis=-1)

    # Create mask based on threshold
    color_mask = (color_diff > threshold)

    return color_mask