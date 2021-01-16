# Read README.md to find detailed instructions about the implementation of each feature.

import cv2
import numpy as np
import argparse
from math import exp

def bright(image,beta,output_image_path):
    if beta == 0:
        factor = -255
    elif beta == 0.5:
        factor = -127
    elif beta == 1.0:
        factor = 0
    elif beta == 1.5:
        factor = 50
    elif beta == 2.0:
        factor = 100
    bright_image = np.zeros(image.shape, image.dtype)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            for cha in range(image.shape[2]):
                bright_image[row, col, cha ] = np.clip(factor + image[row, col, cha],0,255)

    cv2.imshow('Bright Image', bright_image)
    cv2.imwrite(output_image_path,bright_image)

def contrast(image,alpha,output_image_path):

    contrast_image = np.zeros(image.shape, image.dtype)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            for cha in range(image.shape[2]):
                contrast_image[row, col, cha] = alpha * image[row, col, cha]

    cv2.imshow('Contrast Image', contrast_image)
    cv2.imwrite(output_image_path,contrast_image)

def gaussian(x, sigma):
    return exp(-(x**2)/((2*sigma)**2))

def blur(image, sigma,output_image_path):

    '''kernel_radius = 1 generates a 3x3 kernel,
       kernel_radius = 2 generates a 5x5 kernel, and so on...'''

    kernel_radius = 1
    kernel_size =  2 * kernel_radius + 1

    # compute the actual kernel elements
    hkernel = [gaussian(x, sigma) for x in range(kernel_size)]
    vkernel = [x for x in hkernel]
    kernel2d = [[xh * xv for xh in hkernel] for xv in vkernel]

    # normalize the kernel elements
    kernelsum = sum([sum(row) for row in kernel2d])
    kernel2d = [[x / kernelsum for x in row] for row in kernel2d]

    kernel = np.flipud(np.fliplr(kernel2d))
    # convolution output
    blur_image = np.zeros_like(image)

    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2*kernel_radius, image.shape[1] + 2*kernel_radius, 3))
    image_padded[kernel_radius:-kernel_radius, kernel_radius:-kernel_radius,:] = image

    # Loop over every pixel of the image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            for cha in range(image.shape[2]):
                blur_image[row, col, cha] = (kernel * image_padded[row: row + kernel_size, col: col + kernel_size, cha]).sum()

    cv2.imshow('Blurred Image', blur_image)
    cv2.imwrite(output_image_path,blur_image)

def sharpen(image,output_image_path):

    sharpen_image = np.zeros(image.shape, image.dtype)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image_padded = np.zeros((image.shape[0] + 2 , image.shape[1] + 2 , 3))
    image_padded[1:-1, 1:-1, :] = image
    # Loop over every pixel of the image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            for cha in range(image.shape[2]):
                sharpen_image[row, col, cha] = (kernel * image_padded[row: row + 3, col: col + 3, cha]).sum()

    cv2.imshow('Sharpened Image', sharpen_image)
    cv2.imwrite(output_image_path, sharpen_image)

def edge_detection(image,output_image_path):

    edge_image = np.zeros(image.shape, image.dtype)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    image_padded = np.zeros((image.shape[0] + 2 , image.shape[1] + 2 , 3))
    image_padded[1:-1, 1:-1, :] = image
    # Loop over every pixel of the image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            for cha in range(image.shape[2]):
                edge_image[row, col, cha] = (kernel * image_padded[row: row + 3, col: col + 3, cha]).sum()

    cv2.imshow('Edge Detection Image', edge_image)
    cv2.imwrite(output_image_path, edge_image)

def resize(image,nR,nC,output_image_path):

    nrows = image.shape[0]
    ncols = image.shape[1]
    rescaled_image = np.array([[image[int(nrows * r / nR)][int(ncols * c / nC)] for c in range(nC)] for r in range(nR)])
    cv2.imshow('Resized Image', rescaled_image)
    cv2.imwrite(output_image_path, rescaled_image)

def scale_point(image,h,w,output_image_path):

    height,width,channels =image.shape
    new_image=np.zeros((h,w,channels),np.uint8)
    sh=h/height
    sw=w/width
    for i in range(h):
        for j in range(w):
            x=int(i/sh)
            y=int(j/sw)
            new_image[i,j]=image[x,y]
    cv2.imshow('Point/Nearest Neighbor Rescaled Image', new_image)
    cv2.imwrite(output_image_path, new_image)

def scale_bilinear(image,h,w,output_image_path):

    height,width,channels =image.shape
    new_image=np.zeros((h,w,channels),np.uint8)
    value=[0,0,0]
    sh=h/height
    sw=w/width

    for i in range(h):
        for j in range(w):
            x = i/sh
            y = j/sw
            p=(i+0.0)/sh-x
            q=(j+0.0)/sw-y
            x=int(x)-1
            y=int(y)-1
            for k in range(3):
                if x+1<h and y+1<w:
                    value[k]=int(image[x,y][k]*(1-p)*(1-q)+image[x,y+1][k]*q*(1-p)+image[x+1,y][k]*(1-q)*p+image[x+1,y+1][k]*p*q)
            new_image[i, j] = (value[0], value[1], value[2])

    cv2.imshow('Bilinear Rescaled Image', new_image)
    cv2.imwrite(output_image_path, new_image)

def scale_gaussian(image, kernel_radius,output_image_path):

    '''kernel_radius = 1 generates a 3x3 kernel,
       kernel_radius = 2 generates a 5x5 kernel, and so on...'''

    kernel_size =  2 * kernel_radius + 1
    sigma = kernel_radius / 2.

    # compute the actual kernel elements
    hkernel = [gaussian(x, sigma) for x in range(kernel_size)]
    vkernel = [x for x in hkernel]
    kernel2d = [[xh * xv for xh in hkernel] for xv in vkernel]

    # normalize the kernel elements
    kernelsum = sum([sum(row) for row in kernel2d])
    kernel2d = [[x / kernelsum for x in row] for row in kernel2d]

    kernel = np.flipud(np.fliplr(kernel2d))
    # convolution output
    gauss_rescaled_image = np.zeros_like(image)

    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2*kernel_radius, image.shape[1] + 2*kernel_radius, 3))
    image_padded[kernel_radius:-kernel_radius, kernel_radius:-kernel_radius,:] = image

    # Loop over every pixel of the image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            for cha in range(image.shape[2]):
                gauss_rescaled_image[row, col, cha] = (kernel * image_padded[row: row + kernel_size, col: col + kernel_size, cha]).sum()

    cv2.imshow('Gaussian Rescaled Image', gauss_rescaled_image)
    cv2.imwrite(output_image_path,gauss_rescaled_image)

def alpha_composite(foreground, background, alpha,output_image_path):

    foreground = foreground.astype(float)
    background = background.astype(float)
    alpha = alpha.astype(float) / 255

    foreground = alpha * foreground

    background = (1.0 - alpha) * background

    outImage = foreground + background

    cv2.imshow('Composite Image', outImage / 255)
    cv2.imwrite(output_image_path,outImage)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_path',
        type=str,
        help='Specify the path to the image')

    parser.add_argument(
        '--feature',
        type=str,
        help='Specify the feature you want to implement',
        choices = ['brightness','contrast','blur','sharpen','edge_detection','resize','scale_point',
                   'scale_bilinear','scale_gaussian','composite'])

    parser.add_argument(
        '--output_image_path',
        type=str,
        help='Specify the path to the output image')

    parser.add_argument(
        '--alpha',
        type=float,
        help='Enter the contrast factor',
        default = 1.0)

    parser.add_argument(
        '--beta',
        type=float,
        help='Enter the brightness factor',
        default = 0.0)

    parser.add_argument(
        '--sigma',
        type=float,
        help='Enter the value for sigma',
        default=0.0)

    parser.add_argument(
        '--kernel_radius',
        type=int,
        help='Enter the value for kernel_radius',
        default=1)

    parser.add_argument(
        '--new_image_height',
        type=int,
        help='Enter the rescale factor',
        default=100)

    parser.add_argument(
        '--new_image_width',
        type=int,
        help='Enter the rescale factor',
        default=200)

    parser.add_argument(
        '--foreground_image_path',
        type=str,
        help='Specify the path to the foreground image')

    parser.add_argument(
        '--background_image_path',
        type=str,
        help='Specify the path to the background image')

    parser.add_argument(
        '--mask_image_path',
        type=str,
        help='Specify the path to the mask image')

    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    feature = args.feature
    output_image_path = args.output_image_path
    new_image_height = args.new_image_height
    new_image_width = args.new_image_width

    if feature == 'brightness':
        if image is None:
            print('Kindly Specify The Correct Image Path. Incorrect Path: ', args.image_path)
            exit(0)
        cv2.imshow('Original Image', image)
        beta = args.beta
        bright(image,beta,output_image_path)

    elif feature == 'contrast':
        if image is None:
            print('Kindly Specify The Correct Image Path. Incorrect Path: ', args.image_path)
            exit(0)
        cv2.imshow('Original Image', image)
        alpha = args.alpha
        contrast(image,alpha,output_image_path)

    elif feature == 'blur':
        if image is None:
            print('Kindly Specify The Correct Image Path. Incorrect Path: ', args.image_path)
            exit(0)
        cv2.imshow('Original Image', image)
        sigma = args.sigma
        blur(image, sigma,output_image_path)

    elif feature == 'sharpen':
        if image is None:
            print('Kindly Specify The Correct Image Path. Incorrect Path: ', args.image_path)
            exit(0)
        cv2.imshow('Original Image', image)
        sharpen(image,output_image_path)

    elif feature == 'edge_detection':
        if image is None:
            print('Kindly Specify The Correct Image Path. Incorrect Path: ', args.image_path)
            exit(0)
        cv2.imshow('Original Image', image)
        edge_detection(image,output_image_path)

    elif feature == 'resize':
        if image is None:
            print('Kindly Specify The Correct Image Path. Incorrect Path: ', args.image_path)
            exit(0)
        cv2.imshow('Original Image', image)
        resize(image, new_image_height , new_image_width,output_image_path)

    elif feature == 'scale_point':
        if image is None:
            print('Kindly Specify The Correct Image Path. Incorrect Path: ', args.image_path)
            exit(0)
        cv2.imshow('Original Image', image)
        scale_point(image, new_image_height , new_image_width,output_image_path)

    elif feature == 'scale_bilinear':
        if image is None:
            print('Kindly Specify The Correct Image Path. Incorrect Path: ', args.image_path)
            exit(0)
        cv2.imshow('Original Image', image)
        scale_bilinear(image, new_image_height , new_image_width,output_image_path)

    elif feature == 'scale_gaussian':
        if image is None:
            print('Kindly Specify The Correct Image Path. Incorrect Path: ', args.image_path)
            exit(0)
        cv2.imshow('Original Image', image)
        kernel_radius = args.kernel_radius
        scale_gaussian(image, kernel_radius,output_image_path)

    elif feature == 'composite':
        foreground_image_path = args.foreground_image_path
        background_image_path = args.background_image_path
        mask_image_path = args.mask_image_path
        foreground = cv2.imread(foreground_image_path)
        background = cv2.imread(background_image_path)
        mask = cv2.imread(mask_image_path)
        alpha_composite(foreground,background,mask,output_image_path)

    else:
        if image is None:
            print('Kindly Specify The Correct Image Path. Incorrect Path: ', args.image_path)
        print('''Kindly specify the feature you want to implement from the following options: ['brightness','contrast','blur','sharpen','edge_detection','resize','nearest_neighbor_resampling',
                   'bilinear_resampling','composite']''')

    # Wait until user press some key
    cv2.waitKey()

# Read README.md to find detailed instructions about the implementation of each feature.