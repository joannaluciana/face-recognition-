import numpy as np
import face_recognition
import cv2
import os
from datetime import datetime



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
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance (name):
    with open ('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodelistKnown = findEncodings(images)

print('Encoding complete')

cap = cv2.VideoCapture(0)

while True:
    succes, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodelistKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2 - 35),(x2,y2),(0,255,0))
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)





# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon =  face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest =  face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]),(faceLocTest[1], faceLocTest[2]), (255,0,255), 2)




