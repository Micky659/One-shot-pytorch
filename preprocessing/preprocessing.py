import re
import numpy as np
from PIL import Image
import face_detection
import os
import math

from support import notin

class Trainset:

    def __init__(self, path):
        self.path = path
        self.images = []
        self.count = 1

        for subFolder in os.listdir(self.path):
            print("-----------------------------------------------------------------------------------------------")
            print(subFolder)
            for employee in os.listdir(os.path.join(self.path, subFolder)):
                if employee in notin:
                    continue
                for image in os.listdir(os.path.join(self.path, subFolder, employee)):
                    if os.path.getsize(os.path.join(self.path, subFolder, employee, image)) > 1500000:
                        os.remove(os.path.join(self.path, subFolder, employee, image))
                        continue
                    if re.search('script',
                                 os.path.basename(os.path.join(self.path, subFolder, employee, image))) is not None:
                        img1 = os.path.join(self.path, subFolder, employee, image)
                        try:
                            img = Image.open(img1)
                            face1 = self.get_face(img)
                            os.remove(os.path.join(self.path, subFolder, employee, image))
                            path = os.path.join(self.path, subFolder, employee, image)
                            face1.save(path)
                            print("Done: ", self.count)
                            print(os.path.join(self.path, subFolder, employee, image))
                            self.count += 1
                        except:
                            os.remove(os.path.join(self.path, subFolder, employee, image))
                            pass
                    else:
                        img2 = os.path.join(self.path, subFolder, employee, image)
                        try:
                            img = Image.open(img2)
                            face2 = self.get_face(img)
                            os.remove(os.path.join(self.path, subFolder, employee, image))
                            path = os.path.join(self.path, subFolder, employee, image)
                            face2.save(path)
                            print("Done: ", self.count)
                            print(os.path.join(self.path, subFolder, employee, image))
                            self.count += 1
                        except:
                            os.remove(os.path.join(self.path, subFolder, employee, image))
                            pass

    def get_face(self, image):
        try:
            img = np.asarray(image)
            detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
            detection = detector.detect(img)
            print(detection)
            (x1, y1, x2, y2, confidence) = detection[0]
            x1, y1 = abs(x1), abs(y1)
            x1, y1, x2, y2, confidence = round(x1), round(y1), round(x2), round(y2), round(confidence)
            print(x1, y1, x2, y2)
            face = img[y1:y2, x1:x2]

            face_image = Image.fromarray(face)

            return face_image
        except:
            return image


Trainset('/trainset')
