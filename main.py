import sys
import facenet_mv
import time
import cv2

if __name__ == "__main__":
    image = sys.argv[1]
    facenet = facenet_mv.face_ncs()
    tic = time.time()
    processed_image = facenet.process_image(image)
    toc = time.time()
    elapsed = int((toc-tic)*1000)
    print('elapsed_time=', elapsed)
    cv2.imshow('image', processed_image)
    cv2.waitKey()