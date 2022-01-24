import cv2
import matplotlib.pyplot as plt
import pyttsx3
def textToSpeech(txt):
    engine = pyttsx3.init()
    engine.getProperty('rate')
    engine.setProperty('rate', 210)   
    engine.say(txt)
    engine.runAndWait()
config_file="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model="frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(frozen_model,config_file)

classLables = []
file_name='coco.names'
with open(file_name,'rt') as fpt:
    classLables=fpt.read().rstrip('\n').split('\n')
def detect(n):
    img= cv2.imread(n)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    ClassIndex,confidence,bbox = net.detect(img,confThreshold=0.5)
    font_scale = 3
    font=cv2.FONT_HERSHEY_PLAIN
    for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
        cv2.rectangle(img,boxes,(255,0,0),2)
        cv2.putText(img,classLables[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=2)
        print(classLables[ClassInd-1])
        textToSpeech(classLables[ClassInd-1])
    cv2.imshow("Result",img)
