import numpy as np
import face_recognition
import cv2
import os


path = 'ImageAttendance'
images = []
className=[]
myList = os.listdir(path)
print(myList)
for item in myList:
    currImg = cv2.imread(f'{path}/{item}')
    images.append(currImg)
    className.append(os.path.splitext(item)[0])
print (className)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode =  face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodelistKnown = findEncodings(images)

print('Encoding complete')




# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon =  face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest =  face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]),(faceLocTest[1], faceLocTest[2]), (255,0,255), 2)




