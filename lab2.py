'''
import cv2
import numpy as np


def encrypt(image, password):
    encrypted_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            encrypted_image[i, j] = (image[i, j] + ord(password[(i + j) % len(password)])) % 256
    return encrypted_image


def decrypt(encrypted_image, password):
    decrypted_image = np.zeros_like(encrypted_image)
    for i in range(encrypted_image.shape[0]):
        for j in range(encrypted_image.shape[1]):
            decrypted_image[i, j] = (encrypted_image[i, j] - ord(password[(i + j) % len(password)])) % 256
    return decrypted_image


if __name__ == '__main__':
    image = cv2.imread("rx7.jpg", cv2.IMREAD_GRAYSCALE)

    password = "just"
    encrypted_image = encrypt(image, password)
    decrypted_image = decrypt(encrypted_image, password)
    cv2.imwrite("encrypted_image.jpg", encrypted_image)
    cv2.imwrite("decrypted_image.jpg", decrypted_image)


import cv2
import numpy as np


def blue_img(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i, j] == 255:
                image[i, j] = [255, 0, 0]

    return image


if __name__ == '__main__':
    image = cv2.imread("rx7.jpg")
    blue_image = blue_img(image)
    cv2.imwrite("blue_image.jpg", blue_image)


import cv2
import numpy as np

from skimage.metrics import structural_similarity as compare_ssim

def compare_images(image1, image2):
    image1 = cv2.resize(image1, (500, 500))
    image2 = cv2.resize(image2, (500, 500))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    mse = np.mean((gray1 - gray2) ** 2)
    score, diff = compare_ssim(gray1, gray2, full=True)
    return score*100


if __name__ == '__main__':
    image1 = cv2.imread("rx7.jpg")
    image2 = cv2.imread("rover.jfif")
    similarity = compare_images(image1, image2)

    print("Similarity: {:.2f}%".format(similarity))



import cv2

img = cv2.imread("rx7.jpg")
height, width = img.shape[:2]
text = "RX7 Mazda"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 5
font_thickness = 6
text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

x = int((width - text_size[0]) / 2) + 50
y = int(height / 10) +300

text_color = (255, 255, 255)
background_color = (0, 0, 0)

cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

cv2.imwrite("output_image.jpg", img)

import cv2

logo = cv2.imread("ss.JPG")
logo_height, logo_width = logo.shape[:2]
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    x = int(width - logo_width)
    y = int(height - logo_height)

    roi = frame[y:y+logo_height, x:x+logo_width]
    logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(logo_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(logo, logo, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    frame[y:y+logo_height, x:x+logo_width] = dst

    cv2.imshow("last task", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''