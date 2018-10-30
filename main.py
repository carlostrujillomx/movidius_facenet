import sys
import facenet_mv

if __name__ == "__main__":
    image = sys.argv[1]
    facenet = facenet_mv.face_ncs()
    facenet.process_image(image)
    
