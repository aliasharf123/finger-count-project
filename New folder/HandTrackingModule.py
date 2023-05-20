import cv2 as cv2
import mediapipe as mp
import time

class handDectector():
    def __init__(self, mode=False,  max_num_hands = 2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.results = None
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.npDRAW = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.npDRAW.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
        return img


    def findPostions(self,img,handNo=0 ,draw=True):

        lmList =[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)


        return lmList
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDectector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPostions(img)
        if len(lmlist) != 0:
            print((lmlist[4]))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()