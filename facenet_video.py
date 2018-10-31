import sys
import facenet_mv
import time
import cv2

if __name__ == "__main__":
    videofile = sys.argv[1]
    cap = cv2.VideoCapture(videofile)

    facenet = facenet_mv.face_ncs()

    key = 0
    ret = True
    fps = 0
    index = 0
    sum_ = 0
    average_elapsed = 1
    g_tic = time.time()
    while(key!=113 and ret):
        ret, frame = cap.read()
        if ret:
            tic = time.time()
            
            processed_image = facenet.process_video(frame)
            
            elapsed = time.time()-tic#int((time.time() - tic) * 1000)
            sum_+=elapsed
            index+=1
            if index % 30 ==0:
                average_elapsed = sum_/index
            if index == 60:
                index = 0
                sum_ = 0
            fps = int(1/average_elapsed)

            h,w,c=processed_image.shape
            x1 = w-80
            x2 = w 
            y1 = h-40
            y2 = h 
            xs = x1+10
            ys = y1+10
            cv2.rectangle(processed_image, (x1,y1),(x2,y2),(0,0,255),-1)
            cv2.putText(processed_image, str(fps)+'FPS', (xs,ys), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.imshow('video', processed_image)
            key = cv2.waitKey(1)
    
    g_toc = time.time()
    total_elapsed = g_toc - g_tic
    print('total elapsed time =',total_elapsed)



