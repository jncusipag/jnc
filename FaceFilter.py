import numpy as np 
import cv2
import dlib
from scipy.spatial import distance as dst 
from scipy.spatial import ConvexHull
from PIL import Image
from threading import Thread
import face_recognition
from os import listdir
from os.path import isfile, join
from FaceRecognition import Calculate as cal
from FaceRecognition import Place

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

FULL_POINTS = list(range(0,68))
FACE_POINTS = list(range(17,68))

JAWLINE_POINTS = list(range(0,17))
##########Face##########################
RIGHT_EYEBROW_POINTS = list(range(17,22))
LEFT_EYEBROW_POINTS = list(range(22,27))
NOSETIP_POINTS = list(range(31,36))
NOSEBRIDGE_POINTS = list(range(27,31))
RIGHT_EYE_POINTS = list(range(36,42))
LEFT_EYE_POINTS = list(range(42,48))
MOUTH_OUTLINE_POINTS = list(range(48,61))
MOUTH_INNER_POINTS = list(range(61,68))

##############Detection and Prediction#########
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(PREDICTOR_PATH)

##################Start capturing the WebCam#########

face_landmarks = []
process_this_frame = True
video_capture = cv2.VideoCapture(0)
print("FACE FILTER ACTIVATING...")
print("FACE FILTERS ACTIVATED")

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[...,::-1]
	
    if process_this_frame:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		#cv2.imshow("",gray)
        rects = detector(gray,0)

        for rect in rects:
            x = rect.left()
            y = rect.top()
            x1 = rect.right()
            y1 = rect.bottom()
            
            landmarks = np.matrix([[p.x,p.y] for p in predictor(frame,rect).parts()])
            #print (landmarks)
            left_eye = landmarks[LEFT_EYE_POINTS]
            right_eye = landmarks[RIGHT_EYE_POINTS]
            #print(left_eye)
            lip = landmarks[MOUTH_OUTLINE_POINTS]
            face = landmarks[FACE_POINTS]
            beard = landmarks[JAWLINE_POINTS]
            nosetip = landmarks[NOSETIP_POINTS]
            
            lipSize, lipCenter = cal.lip_size(lip)
            noseSize, noseCenter = cal.nosetip_size(nosetip)
            leftEyeSize, leftEyeCenter = cal.eye_size(left_eye)
            rightEyeSize, rightEyeCenter = cal.eye_size(right_eye)
            faceSize, faceCenter = cal.face_size(face)
            beardSize, beardCenter = cal.beard_size(beard)
            Place.nosetip(frame,noseCenter,noseSize)
            #print ("Left - Eye Coordinates")
            #Place.left_eye(frame,leftEyeCenter,leftEyeSize)
            Place.cheeks(frame,beardCenter,beardSize)
            #print ("Right - Eye Coordinates")
            #Place.right_eye(frame,rightEyeCenter,rightEyeSize)
            Place.face(frame,faceCenter,faceSize)
            #Place.lip(frame,lipCenter,lipSize)
            Place.head(frame,faceCenter,faceSize)
        cv2.imshow("Faces with Overlay",frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == ord('q'):
        print("SHUTTING DOWN...")
        break
cv2.destroyAllWindows()













