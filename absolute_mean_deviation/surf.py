import cv2
import matplotlib.pyplot as plt


if(__name__ == "__main__"):

    #read image 
    img = cv2.imread('damar/01_0_islenmis.jpg')

    #instantiate surf descriptor
    surf = cv2.xfeatures2d.SURF_create(400, 4, 3)
    kp, des = surf.detectAndCompute(img,None)
    print "Key points {} and descriptors {}".format(len(kp), len(des))

    # Reduce the number of hessian vectors
    print ("After reducing the number of hessian vectors to a number below 100")
    surf.setHessianThreshold(30000)
    kp, des = surf.detectAndCompute(img,None)
    
    kp = surf.detect(img,None)
    print "Key points {} and descriptors {}".format(len(kp), len(des))
    
    for p in kp:
        print "The keypoint: {}".format(p.pt)
        
    plot =  cv2.drawKeypoints(img,kp,None,(255,0,0),8)
    plt.imshow(plot),plt.show()
    
