#import necessary packages
from scipy.spatial import distance as dst 
from scipy.spatial import ConvexHull
import numpy as np 
import cv2
from os import listdir
from os.path import isfile, join


#create a class for calculating the sizes of the different facial features
class Calculate:
    def eye_size(eye):
	    eyeWidth = dst.euclidean(eye[0],eye[3])
	    hull = ConvexHull(eye)
	    eyeCenter = np.mean(eye[hull.vertices,:],axis=0)
	    eyeCenter = eyeCenter.astype(int)
	    #print (int(eyeWidth),eyeCenter)
	    return int(eyeWidth),eyeCenter
	    
    def nosetip_size(nose):
	    noseWidth = dst.euclidean(nose[0],nose[4])
	    hull = ConvexHull(nose)
	    noseCenter = np.mean(nose[hull.vertices,:],axis=0)
	    noseCenter = noseCenter.astype(int)
	    return int(noseWidth),noseCenter
	    
    def lip_size(lip):
	    lipWidth = dst.euclidean(lip[0],lip[6])
	    hull = ConvexHull(lip)
	    lipCenter = np.mean(lip[hull.vertices,:],axis=0)
	    lipCenter = lipCenter.astype(int)
	    return int(lipWidth),lipCenter
	
    def beard_size(beard):
	    beardWidth = dst.euclidean(beard[2],beard[14])
	    hull = ConvexHull(beard)
	    beardCenter = np.mean(beard[hull.vertices,:],axis=0)
	    beardCenter = beardCenter.astype(int)
	    return int(beardWidth),beardCenter
	
    def face_size(face):
	    faceWidth = dst.euclidean(face[0],face[16])
	    hull = ConvexHull(face)
	    faceCenter = np.mean(face[hull.vertices,:],axis=0)
	    faceCenter = faceCenter.astype(int)
	    return int(faceWidth),faceCenter

class Place:

    def left_eye(frame,eyeCenter,eyeSize):  
	    imgEye = cv2.imread("left_eye.png",-1)
	    orig_mask = imgEye[:,:,3]
	    orig_mask_inv = cv2.bitwise_not(orig_mask)
	    imgEye = imgEye[:,:,0:3]
	    origEyeHeight, origEyeWidth = imgEye.shape[:2]
        
	    eyeSize = int(eyeSize * 1.5)
	    x1 = int(eyeCenter[0,0] - (eyeSize/2))
	    x2 = int(eyeCenter[0,0] + (eyeSize/2))
	    y1 = int(eyeCenter[0,1] - (eyeSize/2))
	    y2 = int(eyeCenter[0,1] + (eyeSize/2))

	    h, w = frame.shape[:2]

	    #check for clipping
	    if x1 < 0:
		    x1=0
	    if y1 < 0:
		    y1=0
	    if x2 >w:
		    x2=w
	    if y2 > h:
		    y2=h

	    #print x1,y1
	    #print x2,y2
	    #re-calculate the size to avoid clipping
	    eyeOverlayWidth = x2 - x1
	    eyeOverlayHeight = y2 - y1

	    #calculate the masks for the overlay
	    eyeOverlay = cv2.resize(imgEye,(eyeOverlayWidth,eyeOverlayHeight),
	        interpolation = cv2.INTER_AREA)
	    mask = cv2.resize(orig_mask,
	        (eyeOverlayWidth,eyeOverlayHeight),interpolation=cv2.INTER_AREA)
	    mask_inv = cv2.resize(orig_mask_inv,
	        (eyeOverlayWidth,eyeOverlayHeight),interpolation=cv2.INTER_AREA)
	
	    #take ROI for the overlay from background, equal to size of the overlay image
	    roi = frame[y1:y2,x1:x2]

	    roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
	
	    roi_fg = cv2.bitwise_and(eyeOverlay,eyeOverlay,mask=mask)
	
	    dst = cv2.add(roi_bg,roi_fg)

	    frame[y1:y2,x1:x2] = dst
	    
    def right_eye(frame,eyeCenter,eyeSize):  
	    imgEye = cv2.imread("right_eye.png",-1)
	    orig_mask = imgEye[:,:,3]
	    orig_mask_inv = cv2.bitwise_not(orig_mask)
	    imgEye = imgEye[:,:,0:3]
	    origEyeHeight, origEyeWidth = imgEye.shape[:2]
        
	    eyeSize = int(eyeSize * 1.5)
	    x1 = int(eyeCenter[0,0] - (eyeSize/2))
	    x2 = int(eyeCenter[0,0] + (eyeSize/2))
	    y1 = int(eyeCenter[0,1] - (eyeSize/2))
	    y2 = int(eyeCenter[0,1] + (eyeSize/2))

	    h, w = frame.shape[:2]

	    #check for clipping
	    if x1 < 0:
		    x1=0
	    if y1 < 0:
		    y1=0
	    if x2 >w:
		    x2=w
	    if y2 > h:
		    y2=h

	    #print x1,y1
	    #print x2,y2
	    #re-calculate the size to avoid clipping
	    eyeOverlayWidth = x2 - x1
	    eyeOverlayHeight = y2 - y1

	    #calculate the masks for the overlay
	    eyeOverlay = cv2.resize(imgEye,(eyeOverlayWidth,eyeOverlayHeight),
	        interpolation = cv2.INTER_AREA)
	    mask = cv2.resize(orig_mask,
	        (eyeOverlayWidth,eyeOverlayHeight),interpolation=cv2.INTER_AREA)
	    mask_inv = cv2.resize(orig_mask_inv,
	        (eyeOverlayWidth,eyeOverlayHeight),interpolation=cv2.INTER_AREA)
	
	    #take ROI for the overlay from background, equal to size of the overlay image
	    roi = frame[y1:y2,x1:x2]

	    roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
	
	    roi_fg = cv2.bitwise_and(eyeOverlay,eyeOverlay,mask=mask)
	
	    dst = cv2.add(roi_bg,roi_fg)

	    frame[y1:y2,x1:x2] = dst
	    
    def lip(frame,lipCenter,lipSize):
        imgLip = cv2.imread("lip.png",-1)
        orig_mask_lip = imgLip[:,:,3]
        orig_mask_inv_lip = cv2.bitwise_not(orig_mask_lip)
        imgLip = imgLip[:,:,0:3]
        lipHeight, lipWidth = imgLip.shape[:2]
        
        lipSize = int(lipSize * 1.5)
        x1 = int(lipCenter[0,0] - (lipSize/2))
        x2 = int(lipCenter[0,0] + (lipSize/2))
        y1 = int(lipCenter[0,1] - (lipSize/4))
        y2 = int(lipCenter[0,1] + (lipSize/4))

        h, w = frame.shape[:2]

        #check for clipping
        if x1 < 0:
            x1=0
        if y1 < 0:
            y1=0
        if x2 >w:
            x2=w
        if y2 > h:
            y2=h

	    #print x1,y1
	    #print x2,y2
	    #re-calculate the size to avoid clipping
        lipOverlayWidth = x2 - x1
        lipOverlayHeight = y2 - y1

	    #calculate the masks for the overlay
        lipOverlay = cv2.resize(imgLip,(lipOverlayWidth,lipOverlayHeight),
            interpolation =cv2.INTER_AREA)
        mask = cv2.resize(orig_mask_lip,
            (lipOverlayWidth,lipOverlayHeight),interpolation=cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv_lip,
            (lipOverlayWidth,lipOverlayHeight),interpolation=cv2.INTER_AREA)

	    #take ROI for the overlay from background, equal to size of the overlay image
        roi = frame[y1:y2,x1:x2]

        roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
	
        roi_fg = cv2.bitwise_and(lipOverlay,lipOverlay,mask=mask)
	
        dst = cv2.add(roi_bg,roi_fg)

        frame[y1:y2,x1:x2] = dst


    def face(frame,faceCenter,faceSize):
        imgFace = cv2.imread("glass.png",-1)
        orig_mask_face = imgFace[:,:,3]
        orig_mask_inv_face=cv2.bitwise_not(orig_mask_face)
        imgFace = imgFace[:,:,0:3]
        faceHeight, faceWidth = imgFace.shape[:2]

        faceSize = int(faceSize * 1.5)
        x1 = int(faceCenter[0,0] - (faceSize/2))
        x2 = int(faceCenter[0,0] + (faceSize/2))
        y1 = int(faceCenter[0,1] - (faceSize/2.3))
        y2 = int(faceCenter[0,1] + (faceSize/15))

        h, w = frame.shape[:2]

        #check for clipping
        if x1 < 0:
            x1=0
        if y1 < 0:
            y1=0
        if x2 >w:
            x2=w
        if y2 > h:
            y2=h

	    #print x1,y1
	    #print x2,y2
	    #re-calculate the size to avoid clipping
        faceOverlayWidth = x2 - x1
        faceOverlayHeight = y2 - y1

	    #calculate the masks for the overlay
        faceOverlay = cv2.resize(imgFace,(faceOverlayWidth,faceOverlayHeight),
            interpolation =cv2.INTER_AREA)
        mask = cv2.resize(orig_mask_face,
            (faceOverlayWidth,faceOverlayHeight),interpolation=cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv_face,
            (faceOverlayWidth,faceOverlayHeight),interpolation=cv2.INTER_AREA)

	    #take ROI for the overlay from background, equal to size of the overlay image
        roi = frame[y1:y2,x1:x2]

        roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
	
        roi_fg = cv2.bitwise_and(faceOverlay,faceOverlay,mask=mask)
	
        dst = cv2.add(roi_bg,roi_fg)

        frame[y1:y2,x1:x2] = dst
        
    def beard(frame,beardCenter,beardSize):
	    imgBeard = cv2.imread("beard.png",-1)
	    orig_mask_beard = imgBeard[:,:,3]
	    orig_mask_inv_beard = cv2.bitwise_not(orig_mask_beard)
	    imgBeard = imgBeard[:,:,0:3]
	    
	    beardSize = int(beardSize * 1.5)
	    x1 = int(beardCenter[0,0] - (beardSize/3))
	    x2 = int(beardCenter[0,0] + (beardSize/3))
	    y1 = int(beardCenter[0,1] - (beardSize/3))
	    y2 = int(beardCenter[0,1] + (beardSize/3))
	    
	    h, w = frame.shape[:2]
	    
	    if x1<0:
	        x1=0
	    if y1<0:
	        y1=0
	    if x2>w:
	        x2=w
	    if y2>h:
	        y2=h
	        
	    beardOverlayWidth = x2 - x1
	    beardOverlayHeight = (y2 - y1)
	    beardOverlay = cv2.resize(imgBeard,
	        (beardOverlayWidth,beardOverlayHeight),interpolation = cv2.INTER_AREA)
	    mask = cv2.resize(orig_mask_beard,
	        (beardOverlayWidth,beardOverlayHeight),interpolation=cv2.INTER_AREA)
	    mask_inv = cv2.resize(orig_mask_inv_beard,
	        (beardOverlayWidth,beardOverlayHeight),interpolation=cv2.INTER_AREA)
	    roi = frame[y1:y2,x1:x2]
	    roi_fg = cv2.bitwise_and(beardOverlay,beardOverlay,mask=mask)
	    roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
	    dst = cv2.add(roi_bg,roi_fg)
	    frame[y1:y2,x1:x2] = dst

    def nosetip(frame,nCenter,nSize):
	    img = cv2.imread("nose.png",-1)
	    orig_mask_nose = img[:,:,3]
	    orig_mask_inv_nose = cv2.bitwise_not(orig_mask_nose)
	    img = img[:,:,0:3]
	    
	    nSize = int(nSize * 1.5)
	    x1 = int(nCenter[0,0] - (nSize/2.1))
	    x2 = int(nCenter[0,0] + (nSize/2.1))
	    y1 = int(nCenter[0,1] - (nSize/2))
	    y2 = int(nCenter[0,1] + (nSize/4))
	    
	    h, w = frame.shape[:2]
	    
	    if x1<0:
	        x1=0
	    if y1<0:
	        y1=0
	    if x2>w:
	        x2=w
	    if y2>h:
	        y2=h
	        
	    nOverlayWidth = x2 - x1
	    nOverlayHeight = (y2 - y1)
	    nOverlay = cv2.resize(img,
	        (nOverlayWidth,nOverlayHeight),interpolation = cv2.INTER_AREA)
	    mask = cv2.resize(orig_mask_nose,
	        (nOverlayWidth,nOverlayHeight),interpolation=cv2.INTER_AREA)
	    mask_inv = cv2.resize(orig_mask_inv_nose,
	        (nOverlayWidth,nOverlayHeight),interpolation=cv2.INTER_AREA)
	    roi = frame[y1:y2,x1:x2]
	    roi_fg = cv2.bitwise_and(nOverlay,nOverlay,mask=mask)
	    roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
	    dst = cv2.add(roi_bg,roi_fg)
	    frame[y1:y2,x1:x2] = dst
	    
    def cheeks(frame,beardCenter,beardSize):
	    imgBeard = cv2.imread("blush.png",-1)
	    orig_mask_beard = imgBeard[:,:,3]
	    orig_mask_inv_beard = cv2.bitwise_not(orig_mask_beard)
	    imgBeard = imgBeard[:,:,0:3]
	    
	    beardSize = int(beardSize * 1.5)
	    
	    x1 = int(beardCenter[0,0] - (beardSize/3.2))
	    x2 = int(beardCenter[0,0] + (beardSize/3.2))
	    y1 = int(beardCenter[0,1] - (beardSize/8))
	    y2 = int(beardCenter[0,1] + (beardSize/40))
	    
	    h, w = frame.shape[:2]
	    
	    if x1<0:
	        x1=0
	    if y1<0:
	        y1=0
	    if x2>w:
	        x2=w
	    if y2>h:
	        y2=h
	        
	    beardOverlayWidth = x2 - x1
	    beardOverlayHeight = (y2 - y1)
	    beardOverlay = cv2.resize(imgBeard,
	        (beardOverlayWidth,beardOverlayHeight),interpolation = cv2.INTER_AREA)
	    mask = cv2.resize(orig_mask_beard,
	        (beardOverlayWidth,beardOverlayHeight),interpolation=cv2.INTER_AREA)
	    mask_inv = cv2.resize(orig_mask_inv_beard,
	        (beardOverlayWidth,beardOverlayHeight),interpolation=cv2.INTER_AREA)
	    roi = frame[y1:y2,x1:x2]
	    roi_fg = cv2.bitwise_and(beardOverlay,beardOverlay,mask=mask)
	    roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
	    dst = cv2.add(roi_bg,roi_fg)
	    frame[y1:y2,x1:x2] = dst
	    
    def head(frame,faceCenter,faceSize):
        imgFace = cv2.imread("hat.png",-1)
        orig_mask_face = imgFace[:,:,3]
        orig_mask_inv_face=cv2.bitwise_not(orig_mask_face)
        imgFace = imgFace[:,:,0:3]
        faceHeight, faceWidth = imgFace.shape[:2]

        faceSize = int(faceSize * 1.5)
        x1 = int(faceCenter[0,0] - (faceSize*1.1))
        x2 = int(faceCenter[0,0] + (faceSize*1.1))
        y1 = int(faceCenter[0,1] - (faceSize*8))
        y2 = int(faceCenter[0,1] + (faceSize/3))

        h, w = frame.shape[:2]

        #check for clipping
        if x1 < 0:
            x1=0
        if y1 < 0:
            y1=0
        if x2 >w:
            x2=w
        if y2 > (h):
            y2=(h)

	    #print x1,y1
	    #print x2,y2
	    #re-calculate the size to avoid clipping
        faceOverlayWidth = x2 - x1
        faceOverlayHeight = y2 - y1

	    #calculate the masks for the overlay
        faceOverlay = cv2.resize(imgFace,(faceOverlayWidth,faceOverlayHeight),
            interpolation =cv2.INTER_AREA)
        mask = cv2.resize(orig_mask_face,
            (faceOverlayWidth,faceOverlayHeight),interpolation=cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv_face,
            (faceOverlayWidth,faceOverlayHeight),interpolation=cv2.INTER_AREA)

	    #take ROI for the overlay from background, equal to size of the overlay image
        roi = frame[y1:y2,x1:x2]

        roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
	
        roi_fg = cv2.bitwise_and(faceOverlay,faceOverlay,mask=mask)
	
        dst = cv2.add(roi_bg,roi_fg)

        frame[y1:y2,x1:x2] = dst



















