import numpy as np
import cv2
from net.vgg19 import VGG19
import time

cap = cv2.VideoCapture(0)
vgg19 = VGG19(vgg19_npy_path='./weights/fine_tune_weight.npy')      # Take time 

def flush_cam_buffer(buffer_size=5):
    foo, boo = cap.read()
    foo, boo = cap.read()
    foo, boo = cap.read()
    foo, boo = cap.read()
    foo, boo = cap.read()

while(True):
    # Capture frame-by-frame
    
    start = time.time()
    flush_cam_buffer() # 0.2s
    ret, img = cap.read()
    label = vgg19.predict(img) # label = 'u', 'd', 'm' # 0.8s
    end = time.time()
    print('{} seconds per image'.format(end-start))
    
    shape = (img.shape[1], img.shape[0])
    if label == 'd':
            cv2.rectangle(img,(0,0),shape,(0,255,0),20)
    elif label == 'u':
        cv2.rectangle(img,(0,0),shape,(0,0,255),20)
    elif label == 'm':
        cv2.rectangle(img,(0,0),shape,(0,100,100),20)

    # Display the resulting frame
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()