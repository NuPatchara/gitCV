import sys
import cv2
import os, os.path
import ColorClustering
import numpy as np
from PIL import Image
from pytesser import *
from sklearn.cluster import KMeans
from sklearn.externals import joblib

def find_most_Color(im, clusters):
    imBodyRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imBodyRGB = imBodyRGB.reshape((imBodyRGB.shape[0] * imBodyRGB.shape[1], 3))
    clt = KMeans(n_clusters=clusters)
    clt.fit(imBodyRGB)
    hist = ColorClustering.centroid_histogram(clt)
    bar = ColorClustering.plot_colors(hist, clt.cluster_centers_)
    return bar

for root, _, files in os.walk('image/'):
    for f in files:
    	imagePath = os.path.join(root, f)
    	if 'tmp' in imagePath: continue
    	cascPath = 'haarcascade_frontalface_default.xml'
        faceCascade = cv2.CascadeClassifier(cascPath)
        image = cv2.imread(imagePath)
        imHeight, imWidth, channels = image.shape
        cv2.putText(image, '%ix%i' %(imWidth,imHeight), (50, 50),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        

        for facesCount, (fx, fy, fw, fh) in enumerate(faces):
            faceCom = imWidth/10 if imWidth>imHeight else imHeight/10
            if faceCom - fh > 50 or faceCom - fh < -30: continue
            cv2.rectangle(image, (fx, fy), (fx + fh, fy + fh), (0, 255, 0), 2)
            cv2.putText(image, 'face: %i' %(facesCount), (fx, fy + fh),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, 'size: %i:%i' %(fw,fh), (fx, fy + fh -30),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, 's.comp: %i ' %(faceCom), (fx, fy + fh -60),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            bodyY = fy+(fh*2)
            bodyX1 = fx - fw / 3
            bodyX1 = 0 if bodyX1 < 0 else bodyX1
            imBody = image[fy + (fh*2):fy + (fh * 5), bodyX1:fx + fw + (fw / 3)]
            cv2.rectangle(image, (bodyX1, fy + (fh*2)), (fx + fw + (fw / 3), fy + (fh * 5)), (0, 255, 255), 2)
            
            gray = cv2.cvtColor(imBody, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
            
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            dicBody = {}
            dicTmp = {}
            dicBest = {}
            for cntCount, cnt in enumerate(contours):
            	[cx,cy,cw,ch] = cv2.boundingRect(cnt)
            	if cv2.contourArea(cnt) > 28 and (fh / 5) < ch < (fh / 2) and cw < (fw / 2) :
            		cv2.rectangle(imBody, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)
            		poin = np.array([cx,cy,cw,ch])
            		numBar = find_most_Color(imBody[cy:cy + ch, cx:cx + cw], 2)
            		dicBody[cntCount] = (poin,numBar[0][0],numBar[0][len(numBar[0]) - 1])

            for cont in dicBody:
            	x = dicBody[cont][0][0]
            	y = dicBody[cont][0][1]
            	h = dicBody[cont][0][3]
            	c1 = dicBody[cont][1]
            	c2 = dicBody[cont][2]
            	clc1 = c1 if c1[0] > c2[0] else c2
            	for c2Count, cont2 in enumerate(dicBody):
            		xl = dicBody[cont2][0][0]
            		yl = dicBody[cont2][0][1]
            		hl = dicBody[cont2][0][3]
            		cl1 = dicBody[cont2][1]
            		cl2 = dicBody[cont2][2]
            		clc2 = cl1 if cl1[0] > cl2[0] else cl2
            		clcl = int(clc1[0]) - int(clc2[0])
            		if yl - 10 < y < yl + 10 and xl - 200 < x < xl + 200 and hl - 5 < h < hl + 5 and abs(clcl) < 10:
            			dicTmp[c2Count] = dicBody[cont2]
            		if len(dicTmp) > len(dicBest) and len(dicTmp) > 1:
            			dicBest = dicTmp
            	dicTmp = {}
            minX = 999999
            maxX = 0
            minY = 999999
            maxY = 0
            for rec in dicBest:
            	[rx, ry, rw, rh] = dicBest[rec][0]
            	if minX > rx:
            		minX = rx
            	if minY > ry:
            		minY = ry
            	if maxY < ry+rh:
            		maxY = ry+rh
            	if maxX < rx+rw:
            		maxX = rx+rw
            if minX <999999 and maxX >0:
            	imgNum = imBody[minY:maxY, minX:maxX]
            	cv2.rectangle(imBody, (minX+2, minY+2), (maxX+2, maxY+2), (255, 0, 255), 2)
            	# cv2.imshow(str(facesCount),imBody)
            	# cv2.imshow('num'+str(facesCount),imgNum)
            	

        cv2.namedWindow("main", cv2.cv.CV_WINDOW_NORMAL)
        imageWinSize = cv2.resize(image, (imWidth/2, imHeight/2))

        while True:
            cv2.imshow("main", imageWinSize)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("w"):
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                sys.exit()