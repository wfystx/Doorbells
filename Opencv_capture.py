import sys
import cv2
#import cv2.cv as cv
import numpy as np

def rotateImage(image, angle):    # normally we dont need this function to rotate pictures
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')  # use classifier to find faces

cap = cv2.VideoCapture("fff.mp4")  # open the
n = 0
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.rotate(gray,1)
    cv2.imshow('f',gray)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    face_cascade.load('haarcascade_frontalface_alt2.xml')  # use opencv classifier

    # Detects objects of different sizes in the input image.
    # The detected objects are returned as a list of rectangles
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #cv2.imshow('hh',frame[y:y+h,x:x+w])
        n = str(n)
        cv2.imwrite("C:/Users/Administrator/Desktop\project/new/Hao1/" +n+'.jpg',gray[y:y+h,x:x+w])
        n=int(n)
        n=n+1
    # Display the resulting frame
        #print(x,y,w,h)
    #cv2.imshow('frame', frame)
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()