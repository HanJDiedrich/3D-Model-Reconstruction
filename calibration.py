import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import camutils

def getImages(filePath):
    images = glob.glob(filePath)
    print(f"Found {len(images)} images")
    # sort to maintain left and right indexes
    return sorted(images)

def intrinsic_Calibration(chessboardDimensions, images, visualize_em):
    # Get interior corners
    rows = chessboardDimensions[0] - 1
    cols = chessboardDimensions[1] - 1
    square_size = chessboardDimensions[2]

    # Set up 3D coordinate grid
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = square_size*np.mgrid[0:cols, 0:rows].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane
 
    num_images_successfully_processed = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])  # note we are assuming all images are the same size
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None) # note the number of squares here

        if ret == True:
            num_images_successfully_processed += 1
            objpoints.append(objp)
            imgpoints.append(corners)

            if visualize_em:
                cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
                cv2.imshow(f"Chessboard Corners {idx+1}", img)
                cv2.waitKey(500)  # Display for 500ms per image
    
        cv2.destroyAllWindows()  # Close all OpenCV windows

    # Call the open cv function that gets the intrinsic camera parameters
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    return K

def extrinsic_Calibration(chessboardDimensions, image, K):
    # Get f and c camera parameters
    my_f = 0.5 * (K[0,0]+K[1,1]) 
    my_c = K[0:2,2] 
    # Get interior corners
    rows = chessboardDimensions[0] - 1
    cols = chessboardDimensions[1] - 1
    square_size = chessboardDimensions[2]
    img = plt.imread(image)

    ret, corners = cv2.findChessboardCorners(img, (cols,rows), None)

    # Format 2D points
    pts2 = corners.squeeze().T

    # Generate 3D points grid
    pts3 = np.zeros((3,rows * cols))
    xx,yy = np.meshgrid(np.arange(cols),np.arange(rows))
    pts3[0,:] = square_size*yy.reshape(1,-1)
    pts3[1,:] = square_size*xx.reshape(1,-1)

    #params_init : 1D numpy.array (dtype=float) initial parameters for camera rotation and translation [rx,ry,rz,tx,ty,tz]
    params_init = np.array([180,0,0,0,0,10])

    cam = camutils.Camera(f=my_f,
                c=np.array([[my_c[0],my_c[1]]]).T,
                t=np.array([[0,0,0]]).T, 
                R=camutils.makerotation(0,0,0))
    
    cam = camutils.calibratePose(pts3,pts2,cam,params_init)

    return cam, pts2, pts3
