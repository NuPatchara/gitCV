import cv2
import sys
import ColorClustering
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from pytesser import *
from pync import Notifier
import re
import sys
import os, os.path

def click_and_crop(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        print 'Click',refPt
        imH,imW,channels = imFace.shape
        sizeDifH = imH/winSize[0]
        sizeDifW = imW/winSize[1]
        for iloca in allCont:
            ix, iy, ix2, iy2 = iloca
            print ix/sizeDifH ,'<', x ,'<', ix2/sizeDifH ,'and', iy/sizeDifH-100 ,'<', y ,'<', iy2/sizeDifH+100
            if ix/sizeDifH < x < ix2/sizeDifH and iy/sizeDifH-100 < y < iy2/sizeDifH+100:
                print '-x: %s y: %s' %(ix,iy)
                lernNum(iloca,imFace)

def lernNum(loca,im):
    x1,y1,x2,y2 = loca
    im = im[y1:y2,x1:x2]
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    samples =  np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]

    if imHeight>imWidth:
        imDiff = imHeight
    else:
        imDiff = imWidth
    
    imDiff2 = 100*faceH/imDiff
    imH, imW, imC = im.shape

    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > imDiff2*5 and h > imH/2:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)
            print(key)
            if key == ord("q"):  # (escape to quit)
                # sys.exit()
                cv2.destroyAllWindows
                break
            if key == 13:
                break
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)
                    # print samples
        # print(' %s/%s' %(cnt,len(contours)))
    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    print "lerning complete"
    cv2.destroyAllWindows()
    generalsamples=open('generalsamples.data','ab')
    np.savetxt(generalsamples,samples)
    generalsamples.close()

    generalresponses=open('generalresponses.data','ab')
    np.savetxt(generalresponses,responses)
    generalresponses.close()

def findNum_fromLern(loca,im):
    minX, minY, maxX, maxY = loca
    samples = np.loadtxt('generalsamples.data',np.float32)
    responses = np.loadtxt('generalresponses.data',np.float32)
    responses = responses.reshape((responses.size,1))

    if imHeight>imWidth:
        imDiff = imHeight
    else:
        imDiff = imWidth
    
    imDiff2 = 100*faceH/imDiff

    model = cv2.KNearest()
    model.train(samples,responses)

    # out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > imDiff2*5 and h > (faceH / 8) and h < (faceH / 2) and w < (faceW / 2):
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
            text = str(int((results[0][0])))
            cv2.putText(imFace, text, (bodyX1+minX+x, bodyY+minY+y - 10),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(imFace,(bodyX1+minX,bodyY+minY),(bodyX1+maxX,bodyY+maxY),(0,0,255),2)
            if [bodyX1+minX,bodyY+minY,bodyX1+maxX,bodyY+maxY] not in allCont:
                allCont.append([bodyX1+minX,bodyY+minY,bodyX1+maxX,bodyY+maxY])

def find_most_Color(im, clusters):
    imBodyRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imBodyRGB = imBodyRGB.reshape((imBodyRGB.shape[0] * imBodyRGB.shape[1], 3))
    clt = KMeans(n_clusters=clusters)
    clt.fit(imBodyRGB)
    hist = ColorClustering.centroid_histogram(clt)
    bar = ColorClustering.plot_colors(hist, clt.cluster_centers_)
    return bar
    
def findNum_digiRec(loca,im):
    for rec in loca:
        [x, y, w, h] = loca[rec][0]
        imgNum = im[y:y + h, x:x + w]
        imgNumGr = cv2.cvtColor(imgNum, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(imgNumGr, (28, 28), interpolation=cv2.INTER_AREA)
        # cv2.imshow('body'+str(faceCount)+'rec '+str(rec),roi)
        roi = cv2.dilate(roi, (3, 3))
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        pText = nbr[0]
        if pText is None:
            pText = -1
        # cv2.putText(im, str(int(nbr[0])), (x, y - 1),cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 100), 3)
        # cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
        # cv2.imshow('body'+str(faceCount),im)
        cv2.putText(imFace, str(int(nbr[0])), (bodyX1+x, bodyY+y - 10),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        # cv2.rectangle(imFace,(bodyX1+x,bodyY+y),(bodyX1+x+w,bodyY+y+h),(0,0,255),2)
        allCont.append([bodyX1+minX,bodyY+minY,bodyX1+maxX,bodyY+maxY])

def findNum_Pytesser(loca,im):
    minX = 999999
    maxX = 0
    minY = 999999
    maxY = 0
    for rec in loca:
        [x, y, w, h] = loca[rec][0]
        if minX > x:
            minX = x
        if minY > y:
            minY = y
        if maxY < y+h:
            maxY = y+h
        if maxX < x+w:
            maxX = x+w
    imgNum = im[minY:maxY, minX:maxX]
    cv2_im = cv2.cvtColor(imgNum,cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('tmpNum%s.jpg'%(faceCount),cv2_im)
    # image_file = 'tmpNum%s.jpg'%(faceCount)
    
    cv2.imwrite('image/tmpNum.jpg',cv2_im)
    image_file = 'image/tmpNum.jpg'
    pil_im = Image.open(image_file)
    text = image_file_to_string(image_file, graceful_errors=True)
    print '======================',text
    if text:
        text = text.strip()
        text = re.sub("[^0-9]", "", text)
        print('XY1: %s,%s XY2: %s,%s' %(minX,maxY,maxX,minY))
        print(text)
        cv2.putText(imFace, text, (bodyX1+minX, bodyY+minY - 10),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(imFace,(bodyX1+minX,bodyY+minY),(bodyX1+maxX,bodyY+maxY),(0,0,255),2)
        allCont.append([bodyX1+minX,bodyY+minY,bodyX1+maxX,bodyY+maxY])
        # cv2.rectangle(im,(minX,minY),(maxX,maxY),(0,0,255),2)
        # cv2.imshow('test',im)
    else:
        im4digi = im[minY:maxY, minX:maxX]
        cv2.rectangle(imFace,(bodyX1+minX,bodyY+minY),(bodyX1+maxX,bodyY+maxY),(0,0,255),2)
        allCont.append([bodyX1+minX,bodyY+minY,bodyX1+maxX,bodyY+maxY])
        # cont4digi = findCont(im4digi)
        # findNum_digiRec(cont4digi,im4digi)
        # lernNum(im4digi)
        findNum_fromLern([minX,minY,maxX,maxY],im4digi)

def findCont(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    samples = np.empty((0, 100))
    responses = []

    dicBody = {}
    dicTmp = {}
    dicBest = {}

    cntCountF = 0
    imRecCont = imBody.copy()
    
    if imHeight>imWidth:
        imDiff = imHeight
    else:
        imDiff = imWidth
    
    imDiff2 = 100*faceH/imDiff

    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        # print cv2.boundingRect(cnt)
        # print str(faceH/8) + ':' + str(faceW/10)
        # print cv2.contourArea(cnt)
        # if cv2.contourArea(cnt) > 50 and h > (faceH / 4) and h < (faceH / 2) and w < (faceW / 4):
        # if faceCount != 3:
        #     break
        # if faceH/10 < h < faceH/2 and faceW/2.5 < w < faceW*1.2:


        
        # if faceH < imHeight/15:
        #     print '========='
        #     if cv2.contourArea(cnt) > 1 and h > (faceH / 10) and h < (faceH / 2) and w < (faceW / 2):
        #         print '----------'
        #         tmp_img = im[y:y + h, x:x + w]
        #         cv2.rectangle(imRecCont, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #         poin = np.array([x,y,w,h])
        #         # numBar = find_most_Color(im[y:y + h, x:x + w], 2)
        #         dicBody[cntCountF] = (poin,numBar[0][0],[])
        #         cntCountF += 1
        #     break
        if cv2.contourArea(cnt) > 50 and h > (faceH / 8) and h < (faceH / 2) and w < (faceW / 2):
            # tmp_img = im[y:y + h, x:x + w]
            # cv2.imshow('cnt'+str(cntCountF),tmp_img)
            cv2.rectangle(imRecCont, (x, y), (x + w, y + h), (0, 0, 255), 2)
            poin = np.array([x,y,w,h])
            numBar = find_most_Color(im[y:y + h, x:x + w], 2)
            dicBody[cntCountF] = (poin,numBar[0][0],numBar[0][len(numBar[0]) - 1])
            cntCountF += 1
    # print('+face: %i cnt: %i dicBody: %s' %(faceCount ,cntCountF,dicBody))
    # print('+face: %i cnt: %i y: %s' %(faceCount ,cntCountF,locaY))
    cv2.imshow(str(faceCount),imRecCont)
    cntCountF = 0
    dTmpCout = 0
    for cont in dicBody:
        y = dicBody[cont][0][1]
        h = dicBody[cont][0][3]
        c1 = dicBody[cont][1]
        c2 = dicBody[cont][2]
        clc1 = []
        if c1[0] > c2[0]:
            clc1 = c1
        else:
            clc1 = c2
        for cont2 in dicBody:
            yl = dicBody[cont2][0][1]
            hl = dicBody[cont2][0][3]
            cl1 = dicBody[cont2][1]
            cl2 = dicBody[cont2][2]
            clc2 = []
            if cl1[0] > cl2[0]:
                clc2 = cl1
            else:
                clc2 = cl2
            clcl = int(clc1[0]) - int(clc2[0])
            # if y > yl - 10 and y < yl + 10 and abs(clcl) < 50:
            # print yl - imDiff2 ,'<', y ,'<', yl + imDiff2 ,'and', hl - imDiff2/4 ,'<', h ,'<', hl + imDiff2/4 ,'and', abs(clcl) ,'<', 50
            # if yl - imDiff2 < y < yl + imDiff2 and hl - imDiff2/4 < h < hl + imDiff2/4:
            if yl - imDiff2 < y < yl + imDiff2 and hl - imDiff2/4 < h < hl + imDiff2/4 and abs(clcl) < 50:
                # print('face: %s cnt: %i c: %s c2: %s dis: %s' %(faceCount,cont2,clc1[0],clc2[0],abs(clcl)))
                dicTmp[dTmpCout] = dicBody[cont2]
                dTmpCout+=1
        if len(dicTmp) > len(dicBest) and len(dicTmp) > 1:
            dicBest = dicTmp
        dicTmp = {}
        # print('face: %i cont: %i dic: %s ' %(faceCount,cont,dicBest))
    return dicBest

for root, _, files in os.walk('image/'):
    for f in files:
        imagePath = os.path.join(root, f)
        # imagePath = 'image/run2.jpg'
        cascPath = 'haarcascade_frontalface_default.xml'
        clf = joblib.load("digits_cls.pkl")
        allCont = []
        
        faceCascade = cv2.CascadeClassifier(cascPath)

        image = cv2.imread(imagePath)
        imHeight, imWidth, channels = image.shape
        winSize = [405,720]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        faceCount = 1
        imFace = image.copy()
        i = 1
        xMax, yMax, wMax, hMax = np.amax(faces, axis=0)

        ##Loop face ----------
        for (x, y, w, h) in faces:
            faceCom = 0
            faceW = w
            faceH = h
            if imWidth>imHeight:
                faceCom = faceW/imWidth
            else:
                faceCom = faceH/imHeight
            if not wMax-(wMax*0.5) < faceW < wMax+(wMax*0.5):
                continue
        #Create Body --------
            cv2.rectangle(imFace, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bodyY = y+(h*2)
            bodyX1 = x - w / 3
            if bodyX1 < 0:
                bodyX1 = 0
            imBody = image[y + (h*2):y + (h * 5), bodyX1:x + w + (w / 3)]
            # cv2.imshow(str(faceCount),imBody)
            # bar = find_most_Color(imBody, 2)
            # cv2.imshow("Color_" + str(faceCount), bar)
            # cv2.imshow('tt'+str(faceCount),imBody)
        ##/Create Body --------
            bestCont = findCont(imBody)
            # print bestCont
            if bestCont:
                findNum_Pytesser(bestCont,imBody)
                # print('-face: %i cnt: %i dicBody: %s' % (faceCount, len(bestCont), bestCont))
            faceCount += 1
        ##/Loop face ----------

        
        cv2.namedWindow("main", cv2.cv.CV_WINDOW_NORMAL)
        imageWinSize = cv2.resize(imFace, (720, 405))
        cv2.setMouseCallback("main", click_and_crop)
        # cv2.imshow("main", imageWinSize)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        while True:
            cv2.imshow("main", imageWinSize)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("w"):
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                sys.exit()





