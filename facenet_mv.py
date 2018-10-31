from mvnc import mvncapi as mvnc 
import numpy as np 
import time
import dlib
import sys
import cv2
import os


class face_ncs():
    def __init__(self):
        self.load_movidius_data()
        self.load_face_processing_files()
        self.frame_counter = 0
        self.image_index = 1
        self.process_flag = False
        self.operation = False
        self.load_encodings()
        self.face_dict = {}
        

    def load_movidius_data(self):
        print('loading movidius graph')
        devices = mvnc.EnumerateDevices()
        num_devices = len(devices)
        if num_devices == 0:
            print('No NCS devices found')
            exit()
        device = mvnc.Device(devices[0])
        device.OpenDevice()

        with open('facenet_celeb_ncs.graph', mode = 'rb') as f:
            graph_on_memory = f.read()
        
        self.graph = device.AllocateGraph(graph_on_memory)
        print('load graph finished')

    def load_face_processing_files(self):
        print('loading face processing data')
        haarcascade = 'haarcascade.xml'
        landmarks = 'landmarks.dat'
        self.face_cascade = cv2.CascadeClassifier(haarcascade)
        self.predictor = dlib.shape_predictor(landmarks)
        print('load face data finished')

    def process_image(self, image):
        self.operation = True
        standard_width = 320
        standard_height = 240
        up_width = 640
        up_height = 480
        image_rsz = cv2.resize(image, (standard_width, standard_height))
        self.upsampled_image = cv2.resize(image, (up_width, up_height))

        ret = self.detect_faces(image_rsz)
        if ret:
            encodings = self.get_encodings(image_rsz)
        return self.upsampled_image

    def process_video(self, frame):
        standard_width = 320
        standard_height = 240
        up_width = 640
        up_height = 480
        image_rsz = cv2.resize(frame, (standard_width, standard_height))
        self.upsampled_image = cv2.resize(frame, (up_width, up_height))

        ret = self.detect_faces(image_rsz)
        if ret:
            encodings = self.get_encodings(image_rsz)
        return self.upsampled_image
        
    def detect_faces(self, image):
        if self.frame_counter % 10 == 0 or self.operation:
            self.process_flag = True
            #print('face_casccade')
            self.faces = self.face_cascade.detectMultiScale(image, 1.2, 2, 0|cv2.CASCADE_SCALE_IMAGE, (10,10), (150,150))
        else:
            self.process_flag = False
        #print('fc:',self.frame_counter)
        self.frame_counter+=1
        if self.frame_counter==60:
            self.frame_counter=0
        #print('faces qty:',len(self.faces))
        if len(self.faces) > 0:
            return True
        else:
            return False

    def get_encodings(self, image):
        encodings = []
              
        for (x1,y1,w,h) in self.faces:
            x2 = x1+w
            y2 = y1+h
            x1_up = int(x1*2)
            y1_up = int(y1*2)
            x2_up = int(x2*2)
            y2_up = int(y2*2)

            if self.process_flag and self.operation:
                cv2.imwrite('face_image'+str(self.image_index)+'.jpg', self.upsampled_image)
                self.image_index+=1
        
            dlib_rect = dlib.rectangle(x1_up,y1_up,x2_up,y2_up)
            shape = self.predictor(self.upsampled_image, dlib_rect)
            rotated = dlib.get_face_chip(self.upsampled_image, shape)
            cv2.rectangle(self.upsampled_image, (x1_up,y1_up), (x2_up,y2_up), (0,0,255), 2)
            
            if self.process_flag:
                self.face_dict.clear()
                inference_encoding = self.run_inference(rotated)
                encodings.append(inference_encoding)
                name = self.face_match(inference_encoding)
                if name != 'unknown':
                    #print('name',name)
                    if name in self.face_dict:
                        self.face_dict[name] = [x1_up,y1_up,x2_up,y2_up]
                    else:
                        self.face_dict.update({name:[x1_up,y1_up,x2_up,y2_up]})
                

            if self.operation:
                cv2.imshow('face_image', rotated)
                key = cv2.waitKey()
                if key == 115:
                    #print(inference_encoding)
                    self.save_encodings(inference_encoding)
                print(key)

        #print('dict',self.face_dict)
        for keys in self.face_dict:
            x1,y1,x2,y2 = self.face_dict[keys]
            x2 = x1+100
            y2 = y1+40
            xs = x1+10
            ys = y1+20
            cv2.rectangle(self.upsampled_image, (x1,y1),(x2,y2),(0,0,255),-1)
            cv2.putText(self.upsampled_image, keys, (xs,ys), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            #print(keys)
        return encodings
    
    
    def save_encodings(self,encodings):
        file = open('faces.txt', 'a')
        for number in encodings:
            file.write(str(number)+' ')
        file.write('\n')
        file.close()
    
    def load_encodings(self):
        self.known_faces = []
        self.known_encodings = []
        file_ = open('faces.txt', 'r')
        for line in file_:
            splitted = line.split(' ')
            name = splitted[0]
            face_properties = splitted[1:129]
            pre_encodings = [float(v) for v in face_properties[:]]
            self.known_faces.append(name)
            self.known_encodings.append(np.array(pre_encodings))
        file_.close()
        print(self.known_faces)
    
    def face_match(self,input_encoding):
        total_diff = 0
        FACE_MATCH_THRESHOLD = 0.1
        name = 'unknown'
        for index in range(0,len(self.known_faces)):
            for in_index in range(0,len(input_encoding)):
                abs_diff = np.square(input_encoding[in_index] - self.known_encodings[index][in_index])
                total_diff += abs_diff
            if total_diff < FACE_MATCH_THRESHOLD:
                name = self.known_faces[index]
                #print('index',index)
            #print('Total Difference=',total_diff)
        #print()
        return name

    def run_inference(self, input_image):
        NETWORK_WIDTH = 160
        NETWORK_HEIGHT = 160
        img = cv2.resize(input_image, (NETWORK_WIDTH, NETWORK_HEIGHT))
        img = img.astype(np.float16)
        self.graph.LoadTensor(img, None)
        output, userobj = self.graph.GetResult()
        if self.operation:
            output = list(output)
        return output


