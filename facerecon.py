import numpy as np
import face_recognition
import cv2

imgElon = face_recognition.load_image_file('ImageBasic/Elon_Musk_front.jpg')
imgElon= cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasic/Elon_Musk_34.jpg')
imgTest= cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon =  face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
print(faceLoc)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)