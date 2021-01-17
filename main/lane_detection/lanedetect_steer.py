import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from PIL import Image, ImageEnhance

################################################################################################
################################         defining pipeline       ###############################
################################################################################################

def grayscale(img):
    """
    Applies the Grayscale transform.
    
    This will return an image with only one color channel."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def brightness_contrast(input_img, contrast, brightness):
    """
    Applies change in Brightness and Contrast.
    
    Works with PIL package, but input image has to be of numpy array."""

    # change Image from np.array to PIL.Image
    img = Image.fromarray(input_img)

    # contrast
    enhancer = ImageEnhance.Contrast(img)
    contrast_img = enhancer.enhance(contrast)

    # brightness
    enhancer = ImageEnhance.Brightness(contrast_img)
    brightness_img = enhancer.enhance(brightness)
    
    # change PIL.Image back to np.array
    img = np.array(brightness_img)

    return img

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transformation."""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices, vertices_car):
    """
    Applies image masks.

    One Mask for the image on the Top and 
    another at the bottom for masking the cars chassis.
    
    For Image mask:
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.

    For car mask:
    Given vertices are the size of black mask.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    # cover also the nose of the cars chassis at bottom of stream
    cv2.fillConvexPoly(masked_image, vertices_car, 0)

    return masked_image

def get_vertices(image, scope):
    """Applies Vertices for later masking of the stream.

    This will return a numpy array 
    depending wether scope is set to car or border."""

    if scope == 'border':
        rows, cols = image.shape[:2]
        bottom_left  = [cols*-0.2, rows]
        top_left     = [cols*0.1, rows*0.4]
        bottom_right = [cols*1.2, rows]
        top_right    = [cols*0.9, rows*0.4] 

        ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        return ver

    elif scope =='car':
        rows, cols = image.shape[:2]
        bottom_left  = [cols*0.2, rows]
        top_left     = [cols*0.4, rows*0.9]
        bottom_right = [cols*0.8, rows]
        top_right    = [cols*0.6, rows*0.9] 

        ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        return ver

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Applies Hough Lines Transformation.

    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn and 
    the lines for later calculations.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    return lines, line_img

def slope_lines(image,lines):
    """Applies slope lines.

    Input ´image´ are therefore the lines from hough transformation.
    
    Those were divided into two categories wether the line ist left or right.
    This will return left and right line with an array of parameters (m, c)
    of the form ´y = m * x + c´."""

    left_lines = [] # Like /
    right_lines = [] # Like \
    for line in lines:
        for x1,y1,x2,y2 in line:

            if x1 == x2:
                pass #Vertical Lines
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1

                if m < 0:
                    left_lines.append((m,c))
                elif m >= 0:
                    right_lines.append((m,c))

    left_line = np.mean(left_lines, axis=0)
    right_line = np.mean(right_lines, axis=0)

    return left_line, right_line

def slope(image, left_line, right_line):
    """Applies slope to an image depending on left and right line.

    This will return a image."""

    img = image.copy()
    poly_vertices = []
    order = [0,1,3,2]
    
    for slope, intercept in [left_line, right_line]:

        # getting complete height of image in y1
        rows, cols = image.shape[:2]
        y1= int(rows)

        # taking y2 upto 60% of actual height or 60% of y1
        y2= int(rows*0.6)

        # we know that equation of line is y=mx +c so we can write it x=(y-c)/m
        x1=int((y1-intercept)/slope)
        x2=int((y2-intercept)/slope)
        poly_vertices.append((x1, y1))
        poly_vertices.append((x2, y2))
        draw_lines(img, np.array([[[x1,y1,x2,y2]]]))
    
    poly_vertices = [poly_vertices[i] for i in order]
    cv2.fillPoly(img, pts = np.array([poly_vertices],'int32'), color = (0,255,0))

    return cv2.addWeighted(image,0.7,img,0.4,0.)

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    Applies lines to an image depending on left and right line.

    This function draws `lines` with `color` and `thickness` directly into image.
    """

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def weighted_img(img, initial_img, α=0.1, β=1., γ=0.):
    """
    Applies weighted image
    
    The result image is computed as follows:
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """

    weighted = cv2.addWeighted(initial_img, α, img, β, γ)
    return weighted

def steer(image, left_line, right_line):
    """
    Calculates Steering factor
    
    The result is the steering value for the model car.
    """

    # calculate center of image
    img_y, img_x = image.shape[:2]
    
    # middle line:
    y1 = int(img_y*0.6) # height of slope
    y2 = int(img_y)

    slope_l = (y1 - left_line[1]) / left_line[0]
    slope_r = (y1 - right_line[1]) / right_line[0]
    slope_with = (slope_r - slope_l)

    x1 = int( ( slope_with / 2) + slope_l )
    x2 = int(img_x/2)

    # if the slope is lower 85 then we assume a false detection and don't want to steer
    if slope_with <100:
        pass
    if slope_with <85: 
        return None
    else:
        steering = (x2 - x1) /100

        return steering

################################################################################################
################################         calling pipeline       ################################
################################################################################################

def lane_detection(image, location='outdoor'):
    if location == 'outdoor':
        picture, canny, steering = lane_finding_pipeline_outdoor(image)
    elif location =='indoor':
        picture, canny, steering = lane_finding_pipeline_indoor(image)

    return picture, canny, steering

# Lane finding Pipeline indoor
def lane_finding_pipeline_indoor(image):

    # Grayscale
    gray_img = grayscale(image)
    # Gaussian Smoothing
    smoothed_img = gaussian_blur(img = gray_img, kernel_size = 3)
    
    ## Canny Edge Detection
    # Calculate good threshold
    med_val = np.median(smoothed_img) 
    lower = int(max(0 ,0.7*med_val))
    upper = int(min(255,1.3*med_val))
    # perform canny edge detection
    canny_img = canny(img = smoothed_img, low_threshold = lower, high_threshold = upper)

    # Mask Image Within a Polygon for each environment and car
    masked_img = region_of_interest(img = canny_img, vertices = get_vertices(image, 'border'), vertices_car = get_vertices(image, 'car'))
    # Hough Transform Lines
    lines, line_img = hough_lines(img = masked_img, rho = 1, theta = np.pi/180, threshold = 20, min_line_len = 20, max_line_gap = 180)
    # draw left and right line
    left_line, right_line = slope_lines(line_img, lines)
    # draw slope between two lines
    slope_weighted_img = slope(line_img, left_line, right_line)
    # add layer with slope lines to original input image
    output = weighted_img(img = slope_weighted_img, initial_img = image, α=0.8, β=1., γ=0.)
    # mask the output image again for better interpretation of results
    canny_mask = region_of_interest(img = canny_img, vertices = get_vertices(image, 'border'), vertices_car = get_vertices(image, 'car'))
    # compute steering advice for car
    steering = steer(image, left_line, right_line)

    return output, canny_mask, steering

# Lane finding Pipeline outdoor
def lane_finding_pipeline_outdoor(image):

    # Grayscale for easier computation
    gray_img = grayscale(image)
    # Change Brightness and Contrast to avoid misclassification caused by ground
    bc_img = brightness_contrast(Input_img = gray_img, contrast = 2, brightness = 0.004)
    # Gaussian Smoothing to get clearness of lines (especialy at noisy grounds like our parklot test ground)
    smoothed_img = gaussian_blur(img = bc_img, kernel_size = 3)    
    
    ## Canny Edge Detection
    # Calculate good threshold
    med_val = np.median(smoothed_img) 
    lower = int(max(0 ,0.7*med_val))
    upper = int(min(255,1.3*med_val))
    # perform canny edge detection
    canny_img = canny(img = smoothed_img, low_threshold = lower, high_threshold = upper)

    # Mask Image Within a Polygon for each environment and car
    masked_img = region_of_interest(img = canny_img, vertices = get_vertices(image, 'border'), vertices_car = get_vertices(image, 'car'))
    # Hough Transform Lines
    lines, line_img = hough_lines(img = masked_img, rho = 0.6, theta = np.pi/180, threshold = 23, min_line_len = 20, max_line_gap = 180)
    # draw left and right line
    left_line, right_line = slope_lines(line_img, lines)
    # draw slope between two lines
    slope_weighted_img = slope(line_img, left_line, right_line)
    # add layer with slope lines to original input image
    output = weighted_img(img = slope_weighted_img, initial_img = image, α=0.8, β=1., γ=0.)
    # mask the output image again for better interpretation of results
    canny_mask = region_of_interest(img = canny_img, vertices = get_vertices(image, 'border'), vertices_car = get_vertices(image, 'car'))
    # compute steering advice for car
    steering = steer(image, left_line, right_line)

    return output, canny_mask, steering


  
