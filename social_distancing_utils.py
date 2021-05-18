# Import modules
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from imutils.object_detection import non_max_suppression

def sort_points(points):
    # Sorts the calibration points so that 1st point corresponds to top-left corner of the image, 2nd point to top-right, etc.
    # Input: 4x2 array of calibration pooints
    # Output: 4x2 array of sorted points

    center = points.sum(axis=0) / 4
    shifted = points - center
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])
    sort_points = points[np.argsort(-theta)]
    
    return sort_points
    
def bird_eye_transform(calibration_rect, width, height):
    # Computes the homography matrix to transform an image to a bird's-eye perspective.
    # Inputs: 
    #   calibration_rec: 4x2 array with the calibration rectangle corners
    #   width: width of the original image
    #   height: height of the original image
    # Output: homography matrix
    
    # Define corners of the image    
    corners = np.array([
        [0, 0], # top-left
        [0, height], # top-right
        [width, height], # bottom-right
        [width, 0]], # bottom-left
        dtype = "float32")

    # Get corners of the calibration rectangle
    (tl, tr, br, bl) = sort_points(calibration_rect)
    
    # STEP 1: calculate initial homography matrix. Results are used to define the dimensions of the target image
    
    # Calculate size of the target image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define target rectangle
    target_rect = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # Calculate homography matrix
    M = cv2.getPerspectiveTransform(calibration_rect, target_rect)
    
    # STEP 2: Calculate final homography matrix
    
    # Reshape corner coordinates for transformation
    coord = np.array([[x, y, 1] for [[x, y]] in corners.reshape(corners.shape[0], 1, 2)]).T

    # Calculate corner coordinates in the bird's-eye plane
    corner_bird = M.dot(coord) # Transform
    corner_bird /= corner_bird[2] # Scale
    corner_bird = corner_bird.T[:,:2] # Reshape

    # Compute the width and height of the new image
    bird_width = int(max(corner_bird[:,0]) - min(corner_bird[:,0]))
    bird_height = int(max(corner_bird[:,1]) - min(corner_bird[:,1]))

    # Compute horizontal and vertical offset
    w_offset = abs(min(0, min(corner_bird[:,0])))
    h_offset = abs(min(0, min(corner_bird[:,1])))
    offset = [w_offset, h_offset]

    # Apply offset to get a centered image
    corner_bird_off = np.array(corner_bird + offset, dtype="float32")
    
    # Calculate homography matrix
    M = cv2.getPerspectiveTransform(corners, corner_bird_off)
    
    return M, bird_width, bird_height
    
def find_feet(image):
    # Finds people in an image and returns the coordinates of the pairs of feet
    # Input: CV2 image object
    # Output: coordinates of pairs of feet
    
    # Use CV2 HOG default people detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Detect people. Return bounding boxes and weights
    (rects, weights) = hog.detectMultiScale(image)
    
    # Apply non-maxima suppression to the bounding boxes
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    boxes = non_max_suppression(rects, probs=None, overlapThresh=0.65 )
      
    # Calculate feet coordinates 
    if len(boxes) > 0:
        x = boxes[:,2] - (boxes[:,2] - boxes[:,0])/2
        y = boxes[:,3] 
        feet = np.array(list(zip(x, y)))
    else:
        feet = []
    
    return feet, boxes
    
def homography(coord, M):
    # Applies an homography transform to a set of coordinates
    # Inputs: 
    #    coord: Mx2 array of coordinates
    #    M: xomography matrix
    # Output: transformed coordinates

    # Check dimensions of input array
    if len(coord) == 0:
        return []

    # Reshape coordinates for transformation
    coord = np.array([[x, y, 1] for [[x, y]] in coord.reshape(coord.shape[0], 1, 2)]).T

    # Calculate feet coordinates in the bird's-eye plane
    hom_coord = M.dot(coord) # Transform
    hom_coord /= hom_coord[2] # Scale
    hom_coord = hom_coord.T[:,:2] # Reshape
    
    return hom_coord
    
def check_distances(points, scale, min_dist):
    # Checks if the distance between a set of points is under a given threshold
    # Inputs:
    #   points: Mx2 array of point coordinates
    #   scale: scale of the image in meters/pixel
    #   min_dist: minimum allowed distance in meters
    # Outputs:
    #   points_too_close: boolean array (True=too close)
    
    # Check dimensions of input array
    if len(points) == 0:
        return [], []
    
    # Calculate distance between each pair of people
    dist = cdist(points, points)*scale

    # Replace diagonal elements
    np.fill_diagonal(dist, np.inf)

    # Calculate min distance for each person
    dist = dist.min(axis=0).reshape(-1,1)

    # Identify points that are too close
    points_too_close = (dist < min_dist).flatten()
    
    return points_too_close
    
def display_boxes(image, boxes, points_too_close):
    # Displays bounding boxes around people, in red if they are too close, green otherwise
    # Inputs: input image, coordinates of bounding boxes and list of points
    # Output: CV2 image object with the bounding boxes
    output_image = image.copy()

    for (box, point_too_close) in zip(boxes, points_too_close):
        if point_too_close:
            color = (0, 0, 255) # Red
        else:
            color = (0, 255, 0) # Green
        cv2.rectangle(output_image, (box[0], box[1]), (box[2], box[3]), color=color, thickness=3)
        
    return output_image
    
def display_points(feet_bird, points_too_close, width, height, bird_width, bird_height):
    # Displays people locations in a bird's-eye view. Points are red if they are too close, green otherwise
    # Inputs: 
    #    feet_bird: coordinates of points in the bird's-eye frame of reference a
    #    points_too_close: list of Boolean values, indicating if a given point is too close or not
    #    height and width: dimensions of the output image
    # Output: CV2 image object with colored points
    
    # Generate blank image
    output_image = 255 * np.ones(shape=[bird_height, bird_width, 3], dtype=np.uint8)

    for (point, point_too_close) in zip(feet_bird, points_too_close):
        if point_too_close:
            color = (0, 0, 255) # Red
        else:
            color = (0, 255, 0) # Green
        cv2.circle(output_image, (int(point[0]), int(point[1])), color=color, radius=30, thickness=-1)
    
    # ONLY FOR DEMO: Crop left side of image, since it's always empty.
    output_image = output_image[:, int(bird_width/2):]
    width = int(width*0.5)
    
    # Resize image to match the input image size
    output_image = cv2.resize(output_image, (width, height))
    
    # Add label to image
    cv2.putText(output_image,"Bird's-eye view", (200, 100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
    
    return output_image
    
def process_image(image, M, scale, min_dist, width, height, bird_width, bird_height):
    # Find feet positions
    feet, boxes = find_feet(image)
    # Transform feet positions
    feet_bird = homography(feet, M)
    # Find people at above/below minimum distance
    points_too_close = check_distances(feet_bird, scale, min_dist)
    # Display results
    bounding_boxes =  display_boxes(image, boxes, points_too_close)
    points =  display_points(feet_bird, points_too_close, width, height, bird_width, bird_height)
    # Stack bounding boxes and bird's-eye perspective horizontally
    results = np.hstack((bounding_boxes, points))
    
    return results