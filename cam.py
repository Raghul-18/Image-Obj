import cv2
from imgdetect import detect,textToSpeech
import time
import os
cam = cv2.VideoCapture("http://192.168.43.1:8080/video")
cam.set(3,1080)
cam.set(4,720)
cam.set(10,150)

cv2.namedWindow("test")

img_counter = 0
textToSpeech("VISION TURNING ON")
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "detect{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        textToSpeech("Detecting Objects") 
        detect(img_name)
        time.sleep(1)
        os.remove(img_name)
        img_counter += 1

cam.release()

cv2.destroyAllWindows()