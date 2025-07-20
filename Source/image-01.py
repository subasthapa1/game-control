import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


class ImageAnalysis:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at path: {path}")

        self.image = cv2.imread(path)
        self.height, self.width = self.image.shape[:2]
        self.channels = self.image.shape[2] if len(self.image.shape) == 3 else 1

    def read_image_to(self, image_type):
        if image_type == 'Color' and self.image.shape[2] == 3:
            img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        elif image_type == 'Gray' and self.image.shape[2] == 3:
            img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        elif self.image.shape[2] == 4:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError("image_type must be 'Color', 'Gray', or 'Unchanged'")
        return img
    def convert_to_1darray(self):
        return self.image.flatten()

    def convert_to_png(self):
        """
        Converting color image to transparent
        :return:
        """
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thres = cv2.threshold(image_gray, 150, 255, cv2.THRESH_BINARY)
        mask_full = cv2.bitwise_not(thres)
        b,g,r = cv2.split(self.image)
        mat = [r,g,b, mask_full]
        image_transparent = cv2.merge(mat)

        return image_transparent

    def create_binary_mask(self):
        return





if __name__ == "__main__":
    loc = os.path.abspath('../Images/subas.png')
    obj = ImageAnalysis(os.path.abspath(loc))
    img = obj.read_image_to('Gray')
    #plt.imshow(img)
    #plt.show()
    transparent_image = obj.convert_to_png()
    plt.imshow(transparent_image)
    plt.show()
