# Writeup for P5:Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Matt, Min-gyu, Kim
---

Steps of this project are the following:  

* Feature (Binned color, Color histogram, Histogram of Oriented Gradients (HOG)) extraction on a labeled training set of images
* Optimize the parameters for feature (Binned color, Color histogram, Histogram of Oriented Gradients (HOG)) extraction
* Train a classifier
* Implement a sliding-window technique and search for vehicles in a single image.
* Build a pipeline for a video stream with rejecing and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[ex_vehicles]: ./figures/car_images.jpg
[ex_non-vehicles]: ./examples/notcar_images.jpg
[hog_car]: ./figures/hog_car.jpg
[hog_notcar]: ./figures/hog_notcar.jpg
[sliding_window]: ./figures/sliding_window.jpg
[output]: ./figures/output.jpg
[det_1]: ./figures/detection_scale_0p8.jpg
[det_2]: ./figures/detection_scale_1p0.jpg
[det_3]: ./figures/detection_scale_1p2.jpg
[det_4]: ./figures/detection_scale_1p8.jpg
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

## Feature Extraction and Training Classifier

### Reading images and labeling
The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are some examples of `vehicle` and `non-vehicle` classes:  

Vehicles
![alt text][ex_vehicles]

Non-vehicles
![alt_text][ex_non-vehicles]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

### Choice of HOG parameters.

At first, I listed up some set of parameters like shown in the below table :  

 | Parameter | Candidates |
 | ------ | ----- |
 | color_space | 'RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb' |
 | orient | 6, 7, 8, 9, 10 |
 | pix_per_cell | 8, 10, 12 |
 | hog_channel | 0, 1, 2, 'ALL' |
 | spatial_sizes | (8, 8), (10, 10), (12, 12) |
 | hist_bins | 12, 14, 16 |

After that, I checked out accuracies of all the combinations of those parametes so that, I could get the list of top 10 best combinations. To make this process quick, I reduced dataset to 1,000 cars and 1,000 non-cars. (The results are stored in `tunning_result_**COLOR_SPACE**.p`) As the result -`report.rpt`-, **YCrCb color space showed better performance compares to the others.** 

I did more experiments with all datasets and YCrCb color space for fine-tunning and then it was possible to fix the parameters as shown below :  

 | Parameter | Candidates |
 | ------ | ----- |
 | color_space | 'YCrCb' |
 | orient | 10 |
 | pix_per_cell | 8 |
 | hog_channel | 0 |
 | spatial_sizes | (16, 16) |
 | hist_bins | 16 |


### Training classifiers using the selected HOG features

I trained [linear SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), [SVM with kernel='poly'](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [Decision Trees](http://scikit-learn.org/stable/modules/tree.html) using the dataset comes from [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/) and chosen parameters for feature extraction. In addition, all the features were scaled to zero mean and unit variance before training.

The following table is its result.  
 |  | Linear SVM | SVM(kernal='poly') | DecisionTree |
 | ------ | ----- | ----- | ----- |
 | Prediction Time (sec) | 0.00643 | 65.454 | 0.00997 |
 | Accuracy | 0.9848 | 0.9924 | 0.9398 |
 
This result shows that **`Linear SVM` is 10,000 times faster than `SVM with kernel='poly'` and has no sigficant difference in accuracy.** 


## Building a pipeline

### Sliding Window Search

The function `find_car` divides an image into small cells having 8x8 pixels. The window which has 8x8 cells (=64x64 pixels) moves 2 cells at a time in an image with extracting features. Multi-scale window is implented by scaling an input image.

![alt text][sliding_window]

Here is the output of my pipeline with sliding window :  

![alt text][image4]
---

#### Reducing false detections
##### Using `decision_function()` of LinearSVM
 `decision_function()` of LinearSVM returns the distance between an input feature and its hyperplane. The `find_car` function considers positive prediction with short distance to hyperplane (< 1.0) as false.  
 
##### Averaging last six `heatmap`s
 `find_car` function stores the positions of positive detections - `heatmap` in each frame of the video in the buffer having the depth of 6. By using the buffer, I could take the mean of last six `heatmap`s and remove 

### Video Implementation

#### Reducing 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here's a [link to my video result](./result_project_video.mp4)

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It was very hard to perfectly remove **false detection**. I guess the following approaches can make it more robust.  
1) Gathering more labeled dataset from the false-detection cases
  Additional training dataset extracted from false-detection cases in the project video could increase the accuracy of classifier in this project.
2) Trying to find optimal parameters for the classifier  
  I've just tunned the parameters for feature extraction. I think that the accuracy of the classifier can be improved by some efforts on tunning the parameters of classifier
