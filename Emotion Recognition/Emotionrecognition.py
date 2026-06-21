from facial_emotion_recognition import EmotionRecognition
import cv2

er = EmotionRecognition(device='cpu')
cam = cv2.VideoCapture(0)

while True:
    _,frame = cam.read()
    new = er.recognise_emotion(frame,return_type='BGR')
    cv2.imshow('Frame',new)
    key = cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()