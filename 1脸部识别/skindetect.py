import cv2
import numpy as np
import time
# cap = cv2.VideoCapture(0)
video_captor = cv2.VideoCapture(0)
video_captor.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
video_captor.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
wisize = (500, 300)
while True:
    ret,img = video_captor.read()

    blur = cv2.GaussianBlur(img,(5,5),0)
    # cv2.imshow('GaussianBlur',blur)
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255])) #red

    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    #Perform morphological transformations to filter out the background noise
    #Dilation increase skin color area
    #Erosion increase skin color area
    dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
    cv2.imshow('dilation',cv2.resize(dilation, wisize))
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)
    cv2.imshow('erosion',cv2.resize(erosion, wisize))
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
    cv2.imshow('dilation_2',cv2.resize(dilation2, wisize))
    filtered = cv2.medianBlur(dilation2,5)
    # cv2.imshow('medianBlur',cv2.resize(filtered, wisize))
    # kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    # dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    # kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    median = cv2.medianBlur(filtered,5)
    # ret,thresh = cv2.threshold(median,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret,thresh = cv2.threshold(median,127,255,0)
    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(img.shape,np.uint8)

    max_area=0

    print('contours', contours)
    if not contours:
        print('no contours')
        continue

    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i

    cnt=contours[ci]

    moments = cv2.moments(cnt)
    if moments['m00']!=0:
                cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                cy = int(moments['m01']/moments['m00']) # cy = M01/M00

    centr=(cx,cy)
    cv2.circle(img,centr,7,(255,255,0),5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'Center',centr,font,1,(0,255,255),2)

    cv2.circle(img,centr,5,[0,0,255],2)
    cv2.drawContours(drawing,[cnt],-1,(0,255,0),2)
    cv2.drawContours(img,[cnt],-1,(0,0,255),10)


    cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    hull = cv2.convexHull(cnt,returnPoints = False)
    hull2 = cv2.convexHull(cnt)
    cv2.drawContours(drawing,[hull2],-1,(255,0,255),2)
    # cv2.drawContours(img,[hull2],-1,(0,0,255),2)

    defects = cv2.convexityDefects(cnt,hull)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        dist = cv2.pointPolygonTest(cnt,centr,True)
        # cv2.line(img,start,end,[0,255,0],2)
        cv2.circle(img,far,5,[0,255,0],3)

    x,y,w,h = cv2.boundingRect(cnt)
    rec_temp = cv2.rectangle(drawing,(x,y),(x+w,y+h),(0,255,0),2)
    key_drawing = drawing[y:y+h, x:x+w]
    # cv2.imshow('key_output',key_drawing)
    cv2.imshow('output',cv2.resize(drawing, wisize))
    cv2.imshow('input',cv2.resize(img, wisize))

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
