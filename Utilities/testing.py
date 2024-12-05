import numpy as np
import matplotlib.pyplot as plt
from . import camutils
from . import decode


def decode(imprefix_color,imprefix,start,threshold_color,threshold):

    nbits = 10
    
    imgs = list()
    imgs_inv = list()
    print('loading',end='')
    for i in range(start,start+2*nbits,2):
        fname0 = '%s%2.2d.png' % (imprefix,i)
        fname1 = '%s%2.2d.png' % (imprefix,i+1)
        print('(',i,i+1,')',end='')
        img = plt.imread(fname0)
        img_inv = plt.imread(fname1)
        if (img.dtype == np.uint8):
            img = img.astype(float) / 256
            img_inv = img_inv.astype(float) / 256
        if (len(img.shape)>2):
            img = np.mean(img,axis=2)
            img_inv = np.mean(img_inv,axis=2)
        imgs.append(img)
        imgs_inv.append(img_inv)
    
    (h,w) = imgs[0].shape
    print('\n')
    
    gcd = np.zeros((h,w,nbits))
    mask = np.ones((h,w))
    for i in range(nbits):
        gcd[:,:,i] = imgs[i]>imgs_inv[i]
        mask = mask * (np.abs(imgs[i]-imgs_inv[i])>threshold)
    
    bcd = np.zeros((h,w,nbits))
    bcd[:,:,0] = gcd[:,:,0]
    for i in range(1,nbits):
        bcd[:,:,i] = np.logical_xor(bcd[:,:,i-1],gcd[:,:,i])
    
    code = np.zeros((h,w))
    for i in range(nbits):
        code = code + np.power(2,(nbits-i-1))*bcd[:,:,i]
    
    #Note:!we need to make use of the color instead of convering to grayscale...
    #...since the object and the background could have the same gray level/brightness but be different colors!
    imc1 = plt.imread(imprefix_color +"%02d" % (0)+'.png')
    imc2= plt.imread(imprefix_color +"%02d" % (1)+'.png')
    color_mask = np.ones((h,w))
    color_mask = color_mask*((np.sum(np.square(imc1-imc2), axis=-1))>threshold_color)

    return code,mask,color_mask


def reconstruct(imprefixL1,imprefixL2,imprefixR1,imprefixR2,threshold1,threshold2,camL,camR):
    '''
    CLh,maskLh,cmaskL = decode(imprefixL1,imprefixL2,0,threshold1,threshold2)
    CLv,maskLv,_ = decode(imprefixL1,imprefixL2,20,threshold1,threshold2)
    CRh,maskRh,cmaskR = decode(imprefixR1,imprefixR2,0,threshold1,threshold2)
    CRv,maskRv,_ = decode(imprefixR1,imprefixR2,20,threshold1,threshold2)
    '''
    # left
    CLh,maskLh = decode.decode_gray(imprefixL2, 0, threshold2)
    CLv,maskLv = decode.decode_gray(imprefixL2, 20, threshold2)
    cmaskL = decode.decode_color(imprefixL1, threshold1)
    
    CRh,maskRh = decode.decode_gray(imprefixR2, 0, threshold2)
    CRv,maskRv = decode.decode_gray(imprefixR2, 20, threshold2)
    cmaskR = decode.decode_color(imprefixR1, threshold1)

    



    CL = CLh + 1024*CLv
    maskL = maskLh*maskLv*cmaskL
    CR = CRh + 1024*CRv
    maskR = maskRh*maskRv*cmaskR

    h = CR.shape[0]
    w = CR.shape[1]

    subR = np.nonzero(maskR.flatten())
    subL = np.nonzero(maskL.flatten())

    CRgood = CR.flatten()[subR]
    CLgood = CL.flatten()[subL]

    _,submatchR,submatchL = np.intersect1d(CRgood,CLgood,return_indices=True)

    matchR = subR[0][submatchR]
    matchL = subL[0][submatchL]

    xx,yy = np.meshgrid(range(w),range(h))
    xx = np.reshape(xx,(-1,1))
    yy = np.reshape(yy,(-1,1))

    pts2R = np.concatenate((xx[matchR].T,yy[matchR].T),axis=0)
    pts2L = np.concatenate((xx[matchL].T,yy[matchL].T),axis=0)
    

    pts3 = camutils.triangulate(pts2L,camL,pts2R,camR)

    return pts2L,pts2R,pts3