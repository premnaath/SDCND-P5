# Vehicle detection project

**Project goals**
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./generated/features_hog.png
[image2]: ./generated/hog_features_1.png
[image3]: ./generated/sliding_window_1.png
[image4]: ./generated/sliding_window_2.png
[image5]: ./generated/detected_objects_1.png
[video1]: ./generated/project_video.mp4

## Contents
The contents of this report are briefly mentione here.

1. Rubic points
2. Discussion
3. Conclusion

## 1. Rubic points
This section mentions each of the rubic criteria and explains how the criteria
is satisfied

### 1.1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.
The submission of this report satisfies this criteria.

### 1.2. Histogram of Oriented Gradients (HOG)
#### 1.2.1. Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.
HOG features are an inherent part in detecting if a given image contains
required objects. HOG features are extracted using `skimage.feature.hog()`
function. This function takes various parameters as input in order to extract
the HOG features. The parameters chosen and their chosen values are as follows,
* orientation = 12
* pixel per cell = 8 x 8
* cell per block = 2

**Color space**
HOG features can be extracted for any color space. Different colorspaces like
RGB, HSV, HSL, LUV and YCrCb were evaluated (based on the training dataset only)
and it was chosen to go ahead with **YCrCb** color space. This color space seemed
to be less computationally intensive and the classification on the test set
for this color space was about 98%.

**Orientation**
More the number of orientations, more will be the direction of gradients
classified in each cell. This parameter was set to **12** to give the optimal
detection of gradients of a car. Increasing this number would mean that more
changes in the gradients are captured and eventually we will start detection
other environmental features, which are unnecessary for this project.

**Pixel per cell**
The dimension in pixels for each cell was tuned to **8x8**. Since the training
images are of size 64x64, cell value which completely covers entire image has
to be chosen. If the size of the cell is too small, then the prominent gradients
in the image might be missed. On the other hand, if the cell size is large,
there is risk loosing information of the gradients. Here sizes smaller and
larger than 8x8 pixels were tested and **8x8** was chosen to give the optimum
performance.

**Cell per block**
This specifies a block of cells over which the gradients are to be normalized.
A larger value here also poses a risk of loosing information. Here different
block sizes were evaluated and **2x2** blocks were chosen as an optimum
parameter.

**Code identification**
The function _get_hog_features()_ shows how the HOG features are extracted. This
function is defined in the `lesson_functions.py` between the lines 18 and 36.

![alt text][image1]

The image below shows all 3 channels of HOG features identified from one of the
test images.
![alt text][image2]


#### 1.2.2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
* Chosen classifier - LinearSVM
* Chosen features - HOG features, spatial features and color histogram
* Length of feature vector - 7872
* Test accuracy - 99%

Linear SVM was chosen as the clasifier. The provided training data was split to
20% as test set. The timeseries of the given GTI dataset was ignored. The
features used for training were HOG features (as explained in the previous
section), spatial features and color histograms.

During the lessons it was observed that the spatial and color features can be
enough for the classifier to predict the test set with around 92% accuracy.
Hence it was decided to use these features for training.

**Spatial features**
Spatial binning of features were made to the color transformed YCrCb image. The
size of the bins were **16x16** pixels.

**Color historgrams**
Each color channel of the transformed YCrCb image were binned and used as
features. There are **16 bins** used for each color channel. These bins are then
concatenated and becomes one feature vector.

**Training**
The the feature vector is a combination of HOG features, spatial features and
color histograms. All together, the combined feature vector is of size 7872.
Feature vector for each car and non car image is extracted and given specific
labels as '1' for cars and '0' for images other than cars.

Now the feature and labels are split randomly for train and test sets using
`train_test_split()`. Once the data is split, it is then normalized using
`StandardScaler()` separately for train and test sets. Scaling is necessary here
as each extracted feature comes with different magnitude. After scaling, the
training is done with the training set. Once completed, the test accuracy is
evaluated with `score()`.

The complete process of training takes a little less than **10s** and the
obtained test accuracy is 99.1%.

**Code identification**
The classifier is trained in the file `searchAndClassify.py`. The features
are extracted using the function _extract_features()_ which extracts the
features such as HOG, spatial features (defined in `lesson_functions.py` between
lines 39-44 in the function _bin_spatial()_) and color histograms (defined in
`lesson_functions.py` between lines 47-57 in the function _color_hist()_).

### 1.3. Sliding window search
#### 1.3.1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?
Once the classifier is trained, it can be used to classify cars and non cars on
a completely different image. Searching for cars can be done by sliding a window
of specified dimensions across the image and extracting features bound by it and
predicting for existance or non-existance of a car.

**Sliding window**
The starting position of the window can be defined and can be moved a specified
number of pixels in x and y direction. It is desirable to have an overlap
between these boxes. The position of each window can be appended to a list and
for each of these windows, the HOG features, spatial features and color
histograms can be extracted. This extracted feature can be used to predict for
existance or non existance of cars using the already trained classifier.

Extracting HOG features as mentioned above can become very compunationally
intensive. There are overlapping regions between the windows where the same
features are extracted again. In order to avoid this, the HOG features are
extracted once for the entire image and then subsampled over teh range of each
sliding window. This makes it computationally less intensive.

**Region of interest**
In order to susbstantially decrease the amount of unnecessary computations, the
lower part of the image, where the road and objects usually appear are chosen
for extraction. This effectively reduces the computation time to less than half
as that for the complete image.

The chosen region was with y-range of 375 and 656 pixels. Since, in the project
video, the ego vehicle drives on the left side of the road, the cars to be
detected always appear on the right side. Hence a threshold in x-direction is
also provided, mainly in order to avoid false positive detections. This x-range
is from the mid-point on the x-axis to the end of x-axis.

**Procedure - Scales 1**
The cars may appear closer or far away from the ego vehicle. Its location
affects its size in the image. Hence it may be necessary for us to search the
image at different scales of sliding window.

Firstly the region of interest is searched with a scale of 2 (which makes the
window size to be 128x128 pixels). Please note that the HOG features are first
extracted for the entire region of interest first and then the sliding window
approach starts. The window slides with 87.5% overlap. For each window position,
the already extracted HOG features are sampled accordingly. The spatial features
and color histograms of the region within the sliding window are also extracted.
This together now makes out feature vector. Now using the already trained
classifier, a prection is made to check the existance of car inside the window.
If the prediction turns to be true, then the window position is just appended
to a list of valid windows. Now the window slides with 87.5% overlap to the next
position and the same procedure repeats.

In addition to the above mentioned sliding window, two more sliding window
techniques are added in order to more confidently detect cars in a frame. The
first extended sliding window technique is added with a window dimension of
98x98 pixels and an overlap of around 66%.

The second extended technique's window dimensions are 48x48 pixels with an
overlap of 83%. The main reason for such a small sliding window dimension is
to detect cars which are far from the go vehicle (as their aparent size would
be small).

The sliding windows defined by the above mentioned three dimensions are combined
together as one list.

**Procedure - Scales 2**
The above procedure returns a set of probable boxes where the car exists. But
this might yield a lot of false positives. In order to confidently detect a car
using the provided classifier, the consolidated region covered by all the boxes
above is again searched for features with a much smaller window. In other words,
the first scaling procedure shortlists the area where cars might exist. Now,
that area can be searched with a smaller box and more overlap, to find more
probable features of a car.

The window slides with a size of 64x64 pixels and now with an overlap of 90%
over an image extracted from the area covered by all the boxes defined from the
first scaling method. The HOG features for the entire sub image is extracted and
subsamplped accordingly. The color histogram and spatial features are computed
and appended to the feature vector and presence of a car is predicted. And,
similar to the above scaling procedure, the valid boxes are appended to a valid
list. The position of these boxes are respective to the extracted image and are
to be transformed to the regions in the full image.

Now the valid boxes from both the scaled sliding window searches are combined
together and cars are found.

**Code identification**
The sliding window function _find_cars()_  and _find_sub_cars()_ is defined in
the file `lesson_functions.py` between the lines 214-386.

The rest of the pipeline procedure is covered in the next rubic point.

![Sliding window][image3]

#### 1.3.2. Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?
From the sliding window procedure, regions of probable car locations are found
and marked by windows of two dimensions. Now a heatmap is constructed from all
the boxes. The overlapping regions of the boxes will correspond to a higher heat
value on the heat map.

Now with thresholding the heatmap, regions of less probability can be
disregarged. The remaining region will directly correspond to the cars in image
and its bounding box.

**Handling false positives**

**Image**
In order to handle false positives, the heatmap is thresholded to a value, high
enough to disregard the false detections, but low enough to consider the actual
car detections.

* Threshold value - 4 min heat detections.

**Optimizing usage of classifier**
The classifier is only as accurate of the features used and the training images
used, hence, it is not perfect. But in order to take advantage of the trained
classifier, two different scales are used. This way, false positives are
filtered and areas of cars are marked with two different dimensions of window,
thus incerasing the heat over the area of the car.

![Pipeline image][image4]

### 1.4. Video implementation
#### 1.4.1. Video link
The classified video can be found in the link mentioned below,

[Project output video](https://drive.google.com/file/d/1E2c404NBOC8MvvOYS74cD0X_zQCJGLzG/view?usp=sharing "My project video")

#### 1.4.2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
False positives may occur anywhere as the classifier is not perfect. Here in
this implementation the false positives are handled in two different ways.

- 1. By using two sizes of sliding window. A car is only marked to exist if both
the defined windows finds valid features.

- 2. By averaging the detected heat over 15 samples. In the video, it might also
happen that the false positives also occur sometimes with high heat. In order to
filter these false detections, the heatmap is averaged over 15 samples. If a
false positive occurs, it doesn't last for long period. Hence by averaging the
heat value over a few samples will get rid of the false positives.

These two methods are used to filter out false positives and only mark the
region where the vehicles exist.

**Code identification**
The handling of false positives is defined in file `findCars.py` between lines
35-126 which explains the usage of two different sliding window with different
overlaps. The actual identification of false positives is done on line 126 where
the thresholding is applied.

![False positives][image5]

### 2. Discussion
This section briefly explains some of the advantages and disadvantages of the
implemented method.

**Advantages**
1. The implemented pipeline find cars in the provided video with fewer false
positives.
2. Usage of two windows with different overlap increases the confidence of the
detected targets.
3. The implemented pipeline only misses the cars in a few frames.
4. The execution time for the given video is around 9 minutes.
5. Since a public dataset is used for training, the detection principle can be
considered robust.

**Disadvantages**
1. The pipeline only detects cars those are on the right side of the ego
vehicle.
2. Averaging procedure decreases the responsiveness of detection.
3. The implementation is very dependant on the classifier that was trained.


### 3. Conclusion
A concept for confidently detecting cars in the provided video is developed and
implemented. The implemented method can be considered to be robust and takes
fairly less computation time.

