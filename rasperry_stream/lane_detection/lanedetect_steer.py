import os
#os.chdir('/Users/Syman/Documents/Studij/Semester05/Seminar/AutonomousCar/lane_detection')

# Do all the relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices, vertices_car):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    #cover also the nose of the car at bottom of stream
    cv2.fillConvexPoly(masked_image, vertices_car, 0)

    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def homography(img):

    # Copy Image for formating
    img_dst = img

    # Four corners of source image
    pts_src = np.array([[0, 0], [1920, 0], [1920, 1080],[0, 1080]]) #OL-OR-UR-UL|X(vertical)-Y(horizontal)

    # Four corners of destination image
    pts_dst = np.array([[-1000, -600], [2920, -600], [1300, 1080],[0, 1080]])

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
    
    # Warp source image to destination based on homography
    img_out = cv2.warpPerspective(img, h, (img.shape[1],img.shape[0]))

    return img_out

def slope_lines(image,lines):

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

def steer(image, left_line, right_line):
    # center of image
    img_y, img_x = image.shape[:2]
    cv2.line(image, (int(img_x/2),0), (int(img_x/2),img_y), color = (0, 255, 0), thickness = 10)
    
    # middle line:
    y1 = int(img_y*0.6) # height of slope
    y2 = int(img_y)
    slope_l = (y1 - left_line[1]) / left_line[0]
    slope_r = (y1 - right_line[1]) / right_line[0]
    slope_with = (slope_r - slope_l)
    x1 = int( ( slope_with / 2) + slope_l )
    x2 = int(img_x/2)

    if slope_with < 100:
        print(slope_with)
    if slope_with <85:
        return None
    else:

        #cv2.line(image, (x1,y1), (x2,y2), color = (0, 0, 255), thickness = 10)

        steering = (x2 - x1) /100
#        print('image'+str(img_x)+' - '+str(img_y))
 #       print('slope_l : '+str(slope_l)+' slope_r: '+str(slope_r)+' slope_middle: '+str(x1)+' img_middle: '+str(x2)+' steering: '+str(steering))

        return steering

def slope(image, left_line, right_line):
    
    img = image.copy()
    poly_vertices = []
    order = [0,1,3,2]
    
    for slope, intercept in [left_line, right_line]:

        #getting complete height of image in y1
        rows, cols = image.shape[:2]
        y1= int(rows) #image.shape[0]

        #taking y2 upto 60% of actual height or 60% of y1
        y2= int(rows*0.6) #int(0.6*y1)

        #we know that equation of line is y=mx +c so we can write it x=(y-c)/m
        x1=int((y1-intercept)/slope)
        x2=int((y2-intercept)/slope)
        poly_vertices.append((x1, y1))
        poly_vertices.append((x2, y2))
        draw_lines(img, np.array([[[x1,y1,x2,y2]]]))
    
    poly_vertices = [poly_vertices[i] for i in order]
    cv2.fillPoly(img, pts = np.array([poly_vertices],'int32'), color = (0,255,0))

    return cv2.addWeighted(image,0.7,img,0.4,0.)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    return lines, line_img

def weighted_img(img, initial_img, α=0.1, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    lines_edges = cv2.addWeighted(initial_img, α, img, β, γ)
    #lines_edges = cv2.polylines(lines_edges,get_vertices(img), True, (0,0,255), 10)
    return lines_edges

def get_vertices(image, scope):

    if scope == 'border':
        rows, cols = image.shape[:2]
        bottom_left  = [cols*-0.2, rows]

        top_left     = [cols*0.1, rows*0.5]
        bottom_right = [cols*1.2, rows]
        top_right    = [cols*0.9, rows*0.5] 

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

################################################################################################
################################         calling pipeline       ################################
################################################################################################

# Lane finding Pipeline
def lane_finding_pipeline(image):
    
    #Grayscale
    gray_img = grayscale(image)
    #Gaussian Smoothing
    smoothed_img = gaussian_blur(img = gray_img, kernel_size = 5)
    #Canny Edge Detection
    canny_img = canny(img = smoothed_img, low_threshold = 50, high_threshold = 150)
    #Masked Image Within a Polygon
    masked_img = region_of_interest(img = canny_img, vertices = get_vertices(image, 'border'), vertices_car = get_vertices(image, 'car'))
    #Hough Transform Lines
    lines, line_img = hough_lines(img = masked_img, rho = 1, theta = np.pi/180, threshold = 20, min_line_len = 20, max_line_gap = 180)
    # draw left and right line
    left_line, right_line = slope_lines(line_img, lines)
    # draw slope between two lines
    slope_weighted_img = slope(line_img, left_line, right_line)
    #Draw lines on edges
    output = weighted_img(img = slope_weighted_img, initial_img = image, α=0.8, β=1., γ=0.)
    output = region_of_interest(img = output, vertices = get_vertices(image, 'border'), vertices_car = get_vertices(image, 'car'))

    steering = steer(image, left_line, right_line)

    return output, steering


################################################################################################
################################          call for test         ################################
################################################################################################


# image = mpimg.imread(f'./Flat_adjusted/Flat_adjusted_09.jpg')

# # plot input image
# fig = plt.figure(figsize=(20, 10),num='TEST')
# ax = fig.add_subplot(1, 2, 1,xticks=[], yticks=[])
# plt.imshow(image)
# ax.set_title("Input Image")
# ax = fig.add_subplot(1, 2, 2,xticks=[], yticks=[])

# picture = lane_finding_pipeline(image)
# plt.imshow(picture)

# # plot also processed image
# ax.set_title("Output Image [Lane Line Detected]") 
# plt.show()

  
