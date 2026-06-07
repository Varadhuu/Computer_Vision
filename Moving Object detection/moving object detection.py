import cv2
import imutils

cam = cv2.VideoCapture(0)
firstframe = None
area = 500
print("Camera opened:", cam.isOpened())

while True:
    _,img = cam.read()
    text = "Normal"
    img = imutils.resize(img,width=1000)

    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussianimg = cv2.GaussianBlur(grayimg,(21,21),0)

    if firstframe is None:
        firstframe = gaussianimg
        continue

    imgdiff = cv2.absdiff(gaussianimg,firstframe)
    threshimg = cv2.threshold(imgdiff,25,255,cv2.THRESH_BINARY)[1]
    threshimg = cv2.dilate(threshimg,None,iterations = 2)
    cnts = cv2.findContours(threshimg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for i in cnts:
        if cv2.contourArea(i) < area:
            continue
        (x,y,w,h) = cv2.boundingRect(i)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        text = "Moving detected"

    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.imshow("camera",img)
            
    
    key = cv2.waitKey(1)
    if key == ord("p"):
        break

cam.release()
cv2.destroyAllWindows()
