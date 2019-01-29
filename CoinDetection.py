import cv2 as cv
import numpy as np

showSteps = True

def display_image(image):
    cv.imshow(cv.namedWindow("image", cv.WINDOW_AUTOSIZE), image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

def resize_image(image, width, height):
    resized_image = cv.resize(image, (width, height))
    if showSteps == True:
        display_image(resized_image)
    return resized_image
    

def RGB_to_greyscale(image):
    greyImage = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    if showSteps == True:
        display_image(greyImage)
    return greyImage

def blur_image(image):
    blurredImage = cv.medianBlur(image, 19) 
    if showSteps == True:
        display_image(blurredImage)
    return blurredImage

def laplace_filter(image):
    laplaceImage = cv.Laplacian(image,-1, ksize=5)
    if showSteps == True:
        display_image(laplaceImage)
    return laplaceImage

def threshold_image(image):
    ret, thresholdImage = cv.threshold(image, 250, 280, cv.THRESH_BINARY)
    if showSteps == True:
        display_image(thresholdImage)
    return thresholdImage

# an implementation of the hough circle transform.
def circular_hough_transform(image, radius_range, search_threshold):
    (rows, columns) = image.shape
    angle = 0
    angle_count = 180
    angle_step_size = int(360/angle_count)
    radius_min = radius_range[0]
    radius_max = radius_range[1]
    circles = []
    circles_radias = []

    # the angles that will be used for calculations.
    sin_angles = np.zeros(angle_count)
    cosin_angles = np.zeros(angle_count)
    
    for index in range(0, angle_count):
        sin_angles[index] = np.sin(angle * np.pi/180) 
        cosin_angles[index] = np.cos(angle * np.pi/180)
        angle = angle + angle_step_size
    
    radius = np.array([i for i in range(radius_min, radius_max)])
    
    for r in radius:

        # the votes for this radius.
        votes = np.full((rows, columns), fill_value=0, dtype=np.uint64)

        # for eaach pixel in the image calculate its votes.
        for x in range(rows):
            for y in range(columns):
                if (image[x][y] == 255):
                    for angle in range(0, 180):
                        a = int(x + round(r * cosin_angles[angle]))
                        b = int(y + round(r * sin_angles[angle]))
                        if a >= 0 and a < rows and b >= 0 and b < columns: 
                            votes[a][b] = votes[a][b] + 1
                else:
                    continue

       # get maximum vote. 
        max_vote = np.amax(votes)

        # print the max vote per radius.
        print('radius: ', r)
        print('max value: ', max_vote)
        
        # make any vote that is below search threshold equal zero.
        if (max_vote > search_threshold):
            votes[votes < max_vote] = 0

            # add the circle to the arrray of circles if it doesn't already exist.
            for x in range(rows):
                for y in range(columns):
                    if (votes[x][y] != 0):
                        if has_duplicate_circle(x, y, r, circles) == False:
                            circles.append((x, y, r))
                            circles_radias.append(r)
    
    return circles, circles_radias

# checks if a circle has an exact duplicate that has already been detected,
# and also check if there are circles that are very close to the circle.
def has_duplicate_circle(circle_x, circle_y, circle_r, circles):
    for circle in circles:
        if circle == (circle_x, circle_y, circle_r):
            return True
        elif circle == (circle_x + 1, circle_y, circle_r) or circle == (circle_x - 1, circle_y, circle_r):
            return True
        elif circle == (circle_x, circle_y + 1, circle_r) or circle == (circle_x, circle_y - 1, circle_r):
            return True
        elif circle == (circle_x, circle_y, circle_r + 1) or circle == (circle_x, circle_y, circle_r - 1):
            return True
        elif circle == (circle_x + 1, circle_y + 1, circle_r) or circle == (circle_x - 1, circle_y - 1, circle_r):
            return True
        elif circle == (circle_x + 1, circle_y, circle_r + 1) or circle == (circle_x - 1, circle_y, circle_r - 1):
            return True
        elif circle == (circle_x, circle_y + 1, circle_r + 1) or circle == (circle_x, circle_y - 1, circle_r - 1):
            return True
        elif circle == (circle_x + 1, circle_y + 1, circle_r + 1) or circle == (circle_x - 1, circle_y - 1, circle_r - 1):
            return True
    
    return False

# read image.
image = np.array(cv.imread("test_image.png", 1))
display_image(image)

# resize the image and then apply canny edge detection.
resized_image = resize_image(image, 400, 400)
grayImage = RGB_to_greyscale(resized_image)
blurredImage = blur_image(grayImage)
laplaceImage = laplace_filter(blurredImage)
thresholdImage = threshold_image(laplaceImage)

detected_circles, detected_circles_radias = np.array(circular_hough_transform(thresholdImage, [50, 70], 150))

detected_circles_radias = np.array(detected_circles_radias)
max_radius = np.amax(detected_circles_radias)
min_radius = np.amin(detected_circles_radias)

total_money = 0.0

# draw the detected circles, the colors depend on the value of the coin.
# red is 1 Egyptian Pound, green is 50 piasters, blue is 25 piasters.
for vertex in detected_circles:
    if (vertex[2] >= min_radius and vertex[2] <= min_radius + 2):
        cv.circle(resized_image, (vertex[1], vertex[0]), vertex[2], (255,0,0), 1)
        cv.rectangle(resized_image, (vertex[1]-2, vertex[0]-2), (vertex[1]-2, vertex[0]-2), (0,0,255), 3)
        total_money = total_money + 0.25
    elif (vertex[2] >= min_radius + 3 and vertex[2] <= max_radius - 3):
        cv.circle(resized_image, (vertex[1], vertex[0]), vertex[2], (0,255,0), 1)
        cv.rectangle(resized_image, (vertex[1]-2, vertex[0]-2), (vertex[1]-2, vertex[0]-2), (0,0,255), 3)
        total_money = total_money + 0.5
    elif (vertex[2] >= max_radius - 2 and vertex[2] <= max_radius):
        cv.circle(resized_image, (vertex[1], vertex[0]), vertex[2], (0,0,255), 1)
        cv.rectangle(resized_image, (vertex[1]-2, vertex[0]-2), (vertex[1]-2, vertex[0]-2), (0,0,255), 3)
        total_money = total_money + 1.0
    else:
        cv.circle(resized_image, (vertex[1], vertex[0]), vertex[2], (255,255,255), 1)
        cv.rectangle(resized_image, (vertex[1]-2, vertex[0]-2), (vertex[1]-2, vertex[0]-2), (0,0,0), 3)

print('total money in picture:', total_money)
print('number of circles in picture:', len(detected_circles))
print(detected_circles)
display_image(resized_image)