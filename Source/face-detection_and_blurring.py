import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

class FaceDetection:
    def __init__(self, img):
        self.image = cv2.imread(img, cv2.IMREAD_COLOR)
        self.height, self.width, self.channels = self.image.shape
        self.net = cv2.dnn.readNetFromCaffe('../model/deploy.prototxt',
                                       '../model/res10_300x300_ssd_iter_140000.caffemodel')
        self.mean = [104, 117, 123]
        self.scale = 1.0
        self.in_width = 300
        self.in_height = 300
        self.detection_threshold = 0.45
        self.font_style = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.face = []

    def create_model(self, frame):

        # Convert the image into a blob format.
        blob = cv2.dnn.blobFromImage(frame, scalefactor=self.scale, size=(self.in_width, self.in_height),
                                     mean=self.mean, swapRB=False, crop=False)
        # Pass the blob to the DNN model.
        self.net.setInput(blob)
    def detect_face_and_blur(self):
        frame = self.image
        image_new = self.image.copy()
        h = frame.shape[0]
        w = frame.shape[1]
        self.create_model(frame)
        # Retrieve detections from the DNN model.
        detections = self.net.forward()
        # Process each detection.
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.detection_threshold:
                # Extract the bounding box coordinates from the detection.
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype('int')
                self.face = self.image[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = 'Confidence: %.4f' % confidence
                label_size, base_line = cv2.getTextSize(label, self.font_style, self.font_scale, self.font_thickness)
                cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (x1, y1), self.font_style, self.font_scale, (0, 0, 0))
                blurred_pic = self.blur_face()
                image_new[y1:y2, x1:x2] = blurred_pic

        return frame, image_new

    def blur_face(self,factor=3):

        h, w = self.image.shape[:2]

        if factor < 1:
            factor = 1  # Maximum blurring
        if factor > 5:
            factor = 5  # Minimal blurring

        # Kernel size.
        w_k = int(w / factor)
        h_k = int(h / factor)

        # Insure kernel is an odd number.
        if w_k % 2 == 0:
            w_k += 1
        if h_k % 2 == 0:
            h_k += 1

        blurred_face = cv2.GaussianBlur(self.face, (int(w_k), int(h_k)), 0, 0)

        return blurred_face

if __name__ == "__main__":
    loc = os.path.abspath('../Images/image1.jpg')
    fc = FaceDetection(loc)
    face, blurred_pic = fc.detect_face_and_blur()
    plt.figure(figsize=[15, 10])
    plt.imshow(face[:, :, ::-1])  # Convert BGR to RGB for Matplotlib
    plt.title("Detected Face(s)")
    plt.show()
    plt.figure(figsize=[15, 10])
    plt.imshow(blurred_pic[:, :, ::-1])  # Convert BGR to RGB for Matplotlib
    plt.title("Blurred Face(s)")
    plt.axis('off')
    plt.show()


