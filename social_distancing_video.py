##################
# Import modules #
##################
import cv2
import numpy as np
from social_distancing_utils import * 

##############
# Parameters #
##############

# Path to input and output videos
in_path = 'town_centre.avi' 
out_path = 'output.avi' 
# Calibration mode
cal_path = 'calibration.png'
cal_mode = False
# Delay between frames. Increase if video plays too fast.
delay = 1
# Minimum distance in meters
min_dist = 2
# Image scale (in meters/pixel) 
scale = 6/600 

# X-Y coordinates of the rectangle used for calibration
calibration_rect = np.array([ 
    [1200, 169],
    [1582, 214],
    [942, 871],
    [298, 732]],
    dtype = "float32")

##################
# Initialization #
##################

# Open input video stream
cap = cv2.VideoCapture(in_path)

# Compute video resolution and FPS
width  = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Open output video stream
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_path, fourcc, fps, (int(width*1.5), height))

# Compute homography matrix to generate bird's-eye view
M, bird_width, bird_height = bird_eye_transform(calibration_rect, width, height)


#################
# Process video #
#################

while(cap.isOpened()):
    # Read video frame
    ret, frame = cap.read() 
    # If a frame was returned, process it
    if ret==True:    
        # In calibration mode, return first video frame and exit
        if cal_mode:
            cv2.imwrite(cal_path,frame)
            break
        # Process frame
        results = process_image(frame, M, scale, min_dist, width, height, bird_width, bird_height)
        # Display output 
        cv2.imshow('Frame', results)
        # Write to output video
        out.write(results)
        # Press q key to exit       
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    # Break when no frames are returned
    else:
        break

# Close video stream
cap.release()
cv2.destroyAllWindows()