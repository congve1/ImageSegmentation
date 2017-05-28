import cv2
import numpy as np
import matplotlib.pyplot as plt
import SaliencyRC
import segmentImage
def processImage(filename):
    img3i = cv2.imread(filename)
    img3f = img3i.astype(np.float32)
    img3f *= 1. / 255
    sal = SaliencyRC.GetHC(img3f)
    #idxs = np.where(sal < (sal.max() + sal.min()) / 4.96)
    #img3i[np.where(sal < (sal.max() + sal.min())/4)] = 0
    idxs = np.where(sal >= (sal.max() + sal.min()) / 2)
    I = [np.min(idxs[0]),np.max(idxs[0])]
    J = [np.min(idxs[1]),np.max(idxs[1])]
    imgBBox = img3i.copy()
    imgBBox[I[0]:I[0]+1,J[0]:J[1]+1] = [0,0,255]
    imgBBox[I[0]:I[1]+1,J[1]:J[1]+1] = [0,0,255]
    imgBBox[I[1]:I[1]+1,J[0]:J[1]+1] = [0,0,255]
    imgBBox[I[0]:I[1]+1,J[0]:J[0]+1] = [0,0,255]
    resImg = np.zeros(img3i.shape,img3i.dtype)
    resImg[I[0]:I[1],J[0]:J[1]] = img3i[I[0]:I[1],J[0]:J[1]]
    resImg[np.where(sal < (sal.max() + sal.min())/ 4)] = 0
    sal *= 255
    cv2.imwrite("{}process.jpg".format(filename),resImg)
    cv2.imwrite("{}bbox.jpg".format(filename),imgBBox)
    cv2.imwrite("{}hc.jpg".format(filename),sal.astype(np.int16))



def processImage1(filename):
    img3i = cv2.imread(filename)
    img3f = img3i.astype(np.float32)
    img3f *= 1. /255
    regNum, regIdx1i = segmentImage.SegmentImage(img3f, None, 0.5,200,50)
    bdgReg1u = SaliencyRC.GetBorderReg(regIdx1i,regNum,0.02,0.4)
    bdgIdxs = np.where(bdgReg1u == 255)
    img3i[bdgIdxs] = 0
    cv2.imwrite("{}process.jpg".format(filename),img3i)

