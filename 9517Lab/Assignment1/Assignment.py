import cv2
import numpy as np


def max_filtered(image, n):
    # get the height and width of the input image
    height = image.shape[0]
    width = image.shape[1]
    # create a new image with the same size as the input image
    new_image = np.zeros_like(image)
    # set the padding size
    p = n // 2
    # add padding according to the kernel size and get a image with padding
    image_padded = cv2.copyMakeBorder(image, p, p, p, p, cv2.BORDER_DEFAULT)
    for i in range(p, height + p):
        for j in range(p, width + p):
            new_image[i - p][j - p] = np.max(image_padded[i - p:i + p + 1, j - p:j + p + 1])
    return new_image


def min_filtered(image, n):
    # get the height and width of the image
    height = image.shape[0]
    width = image.shape[1]
    # create a new image with the same size as the input image
    new_image = np.zeros_like(image)
    # set the padding size
    p = n // 2
    # add padding according to the kernel size and get a image with padding
    image_padded = cv2.copyMakeBorder(image, p, p, p, p, cv2.BORDER_DEFAULT)
    for i in range(p, height + p):
        for j in range(p, width + p):
            new_image[i - p][j - p] = np.min(image_padded[i - p:i + p + 1, j - p:j + p + 1])
    return new_image


if __name__ == '__main__':
    image_P = cv2.imread('Particles.png', 0)
    image_C = cv2.imread('Cells.png', 0)
    # M can be changed to 1
    M = 0
    # Task 1 and Task 2
    if M == 0:
        N = 11
        image_A = max_filtered(image_P, N)
        image_B = min_filtered(image_A, N)
        image_O = np.zeros_like(image_P)
        for i in range(image_P.shape[0]):
            for j in range(image_P.shape[1]):
                image_O[i][j] = image_P[i][j].astype(np.int32) - image_B[i][j].astype(np.int32) + 255
        image_O = image_O.astype(np.uint8)
    # Task 3
    elif M == 1:
        N = 29
        image_A = min_filtered(image_C, N)
        image_B = max_filtered(image_A, N)
        image_O = np.zeros_like(image_C)
        for i in range(image_C.shape[0]):
            for j in range(image_C.shape[1]):
                image_O[i][j] = image_C[i][j] - image_B[i][j]
    cv2.imwrite('image_A.png', image_A)
    cv2.imwrite('image_B.png', image_B)
    cv2.imwrite('image_O.png', image_O)
    cv2.waitKey(0)