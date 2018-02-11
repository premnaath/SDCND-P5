import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import cv2
from lesson_functions import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

# Global variable declaration
previous_centroid = []
counter = 0
heat_averaged = []
mean_over_samples = 15

# Get classifier data
classifier_file = 'classifier.p'
with open(classifier_file, mode='rb') as f:
    classifier = pickle.load(f)

svc, X_scaler= classifier['svc'], classifier['X_scaler']

# Define a main function to start the vehicle detection procedure.
def findCars(image):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    orient = 12
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (16, 16)
    hist_bins = 16
    ystart = 375
    ystop = 656
    scale = 2

    # Call the initial function to find region of interest.
    # This function uses a larger sliding window and lesser overlap
    # to narrow down the area for vehicle detection.
    out_img_full, out_img1, hot_windows_shifted = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                                            cell_per_block, spatial_size, hist_bins)

    # A procedure to increase confidence in detection.
    # Can help in removing false positives.
    hot_windows_shifted.extend(hot_windows_shifted)

    # Run the sliding window with smaller size only if the
    # larger window detects car features.
    if len(hot_windows_shifted) > 0:
        # Find the area of interest from the larger sliding window technique.
        minvals = np.min(hot_windows_shifted, axis=0)
        maxvals = np.max(hot_windows_shifted, axis=0)

        # Widen the search area by 30 pixels to allow better feature detection.
        xmin = minvals[0][0] - 30
        ymin = minvals[0][1]
        xmax = min(maxvals[1][0] + 30, image.shape[1])
        ymax = maxvals[1][1]

        extracted_image = image[ymin:ymax, xmin:xmax, :]

        # Run sliding window with smaller size and larger overlap.
        extracted, hot_windows = find_sub_cars(extracted_image, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                               spatial_size, hist_bins)

        # Empty list to append windows to.
        hot_windows_offset = []

        # Offset the windows
        to_add = [(xmin, ymin), (xmin, ymin)]
        for a,b in hot_windows:
            for dx,dy in to_add:
                added = [(a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy)]
                hot_windows_offset.append(added)

        # Append the smaller windows to the detected larger windows
        hot_windows_shifted.extend(hot_windows_offset)

    # Draw the detected boxes.
    windowed = draw_boxes(np.copy(image), hot_windows_shifted, color=(0, 0, 255), thick=3)

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows_shifted)

    # Global variables access
    global heat_averaged
    global counter
    global mean_over_samples

    # Create a history heatmap.
    if (len(hot_windows_shifted) > 0) & (np.sum(heat) > 0):
        if len(heat_averaged) == 0:
            heat_averaged = heat
            counter += 1
        else:
            counter += 1
            if counter <= mean_over_samples:
                heat_averaged = np.dstack((heat_averaged, heat))
            else:
                index = counter % mean_over_samples
                heat_averaged[:, :, index] = heat

    # Average the heatmap.
    if counter > 1:
        heatavg = np.mean(heat_averaged, axis=2)
    else:
        heatavg = heat

    # Apply threshold to the history heatmap.
    heat = apply_threshold(heatavg, 4)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    # Draw the labeled boxes.
    draw_img, centroids = draw_labeled_bboxes(np.copy(image), labels)

    # Create image for visualization.
    heatmap_gray = heatmap.astype(np.uint8) * 200
    heatmap_gray = np.clip(heatmap_gray, 0, 255)
    heatmap_3ch = np.dstack((heatmap_gray, heatmap_gray, heatmap_gray*0))

    weightedimage = cv2.addWeighted(image, 1, heatmap_3ch, 1, 0)
    tobe_stacked = np.hstack((windowed, weightedimage))

    # Stack images.
    tobe_stacked = cv2.resize(tobe_stacked, (tobe_stacked.shape[1]//3, tobe_stacked.shape[0]//3))

    # Overlay the stacked images.
    draw_img[0:tobe_stacked.shape[0], 0:tobe_stacked.shape[1], :] = tobe_stacked

    return draw_img

#######################################################################################################################

# Option specifier to run on images or video.
runimage = True

if runimage:
    image = mpimg.imread('./CarND-Vehicle-Detection/test_images/test1.jpg')

    drawn_image = findCars(image)

    plt.imshow(drawn_image)
    plt.title("Vehicle detection")
    plt.show()

else:
    white_output = 'project_video_output.mp4'
    clip1 = VideoFileClip("./CarND-Vehicle-Detection/project_video.mp4")
    # clip1 = VideoFileClip("./CarND-Vehicle-Detection/project_video.mp4").subclip(10, 12)
    # clip1 = VideoFileClip("./CarND-Vehicle-Detection/project_video.mp4").subclip(23.5, 27)

    white_clip = clip1.fl_image(findCars)
    white_clip.write_videofile(white_output, audio=False)


