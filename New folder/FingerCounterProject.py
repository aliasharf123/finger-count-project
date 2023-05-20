import cv2
import mediapipe
import os
import time
import  HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


folderPath = 'FingerImages'
myList = os.listdir(folderPath)
print(myList)
overlayList = []
pTime= 0
cTime=0

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

detector = htm.handDectector()

tip = [4, 8, 12, 16,20]

while True:
    success, img =cap.read()
    img =detector.findHands(img)
    lmList = detector.findPostions(img)
    totalFingers = 0
    if len(lmList) !=0 :
        figers=[]

        #Thumb
        if lmList[tip[0]][1] > lmList[tip[0] - 1][1]:
            figers.append(1)
        else:
            figers.append(0)

        for id in range(1,   5):
            if lmList[tip[id]][2] < lmList[tip[id]-2][2]:
                figers.append(1)
            else:
                figers.append(0)
        totalFingers = figers.count(1)
        cv2.putText(img, str(int(totalFingers)), (550, 400), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
    h, w, c = overlayList[totalFingers].shape
    img[0:h, 0:w] = overlayList[totalFingers]
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (500, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)