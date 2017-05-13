import cv2
import numpy as np
import matplotlib.pyplot as plt

def processImage(filename):
    img = grabcut(filename)
    fig = plt.figure()
    ax1 = fig.add_subplot(111,aspect="equal")

    plt.imshow(img)
    plt.savefig('{}process.jpg'.format(filename))
    plt.close()

def grabcut(filename):
    tmp = cv2.imread(filename)
    img = cv2.resize(tmp,(250,200),interpolation=cv2.INTER_CUBIC)
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (0,0,200,200)
    mask = cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)[0]
    mask2 = np.where((mask == 2) | (mask==0),0,1).astype('uint8')
    return img * mask2[:,:,np.newaxis]