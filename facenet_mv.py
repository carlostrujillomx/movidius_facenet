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

    def load_movidius_data(self):
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

    def load_face_processing_files(self):
        haarcascade = 'haarcascade.xml'
        landmarks = 'landmarks.dat'
        self.face_cascade = cv2.CascadeClassifie(haarcascade)
        self.predictor = dlib.shape_predictor(landmarks)

    def process_image(self, image):
        standard_width = 320
        standard_height = 240

        image_rsz = cv2.resize(image, (standard_width, standard_height))
        ret = self.detect_faces(image_rsz)
        if ret:
            encodings = self.get_encodings(image_rsz)
            print(encodings)

    def detect_faces(self, image):
        self.faces = self.face_cascade.detectMultiScale(image, 1.1, 3)
        if len(self.faces) > 0:
            return True
        else:
            return False

    def get_encodings(self, image):
        encodings = []
        for (x1,y1,w,h) in self.faces:
            x2 = x1+w
            y2 = y1+h
            dlib_rect = dlib.rectangle(x1,y1,x2,y2)
            shape = self.predictor(image, dlib_rect)
            rotated = dlib.get_face_chip(image, shape)
            encodings.append(self.run_inference(rotated))

        return encodings

    def run_inference(self, input_image):
        NETWORK_WIDTH = 160
        NETWORK_HEIGHT = 160
        img = cv2.resize(input_image, (NETWORK_WIDTH, NETWORK_HEIGHT))
        img = img.astype(np.float32)
        self.graph.LoadTensor(img, None)
        output, userobj = self.graph.GetResutl()

        return output


