# Writeup for P5:Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Matt, MinGyu, Kim
---

Steps of this project are the following:  

* Feature (Binned color, Color histogram, Histogram of Oriented Gradients (HOG)) extraction on a labeled training set of images
* Optimize the parameters for feature (Binned color, Color histogram, Histogram of Oriented Gradients (HOG)) extraction
* Train a classifier
* Implement a sliding-window technique and search for vehicles in a single image.
* Build a pipeline for a video stream with rejecing and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
* Video implementation

#### All the steps are implemented in `Vehicle_Detection.ipynb`

[//]: # (Image References)
[ex_vehicles]: ./figures/car_images.jpg
[ex_non-vehicles]: ./figures/notcar_images.jpg
[hog_car]: ./figures/hog_car.jpg
[hog_notcar]: ./figures/hog_notcar.jpg
[sliding_window]: ./figures/sliding_window.jpg
[output]: ./figures/output.jpg
[det_1]: ./figures/detection_scale_0p8.jpg
[det_2]: ./figures/detection_scale_1p0.jpg
[det_3]: ./figures/detection_scale_1p2.jpg
[det_4]: ./figures/detection_scale_1p8.jpg
[label_map]: ./figures/label_map.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
#### [Here](https://review.udacity.com/#!/rubrics/513/view) I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

## Feature Extraction and Training Classifier

### Reading images (Step 1-1 in `Vehicle_Detection.ipynb`)
The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are some examples of `vehicle` and `non-vehicle` classes:  

Vehicles
![alt text][ex_vehicles]

Non-vehicles
![alt_text][ex_non-vehicles]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog_car]
![alt text][hog_notcar]

### Choice of HOG parameters (Step 1-2 in `Vehicle_Detection.ipynb`)

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

---

## Training classifiers using the selected HOG features (Step 2 in `Vehicle_Detection.ipynb`)  

I trained [linear SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), [SVM with kernel='poly'](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [Decision Trees](http://scikit-learn.org/stable/modules/tree.html) using the dataset comes from [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/) and chosen parameters for feature extraction. In addition, all the features were scaled to zero mean and unit variance before training.

The following table is its result. 

 |  | LinearSVM | SVM (kernal='poly') | DecisionTree |
 | ------ | ----- | ----- | ----- |
 | Prediction Time (sec) | 0.00643 | 65.454 | 0.00997 |
 | Accuracy | 0.9848 | 0.9924 | 0.9398 |
 
This result shows that **`Linear SVM` is 10,000 times faster than `SVM with kernel='poly'` and has no sigficant difference in accuracy.** 

---

## Building a pipeline

### Sliding Window Search by sub-sampling (Step 3-1 in `Vehicle_Detection.ipynb`)  

The function `find_car` divides an image into small cells having 8x8 pixels. The window which has 8x8 cells (=64x64 pixels) moves 2 cells at a time in an image with extracting features. Multi-scale window is implented by scaling an input image. This sub-sampling method operates faster than real sliding-window method.

Here are the scaled images with all the possible windows :  
![alt text][sliding_window]

The followings are the results of detection for various scaling :  
![alt text][det_1]
![alt text][det_2]
![alt text][det_3]
![alt text][det_4]


### Reducing false detections (Step 3-1 in `Vehicle_Detection.ipynb`)  
#### Using `decision_function()` of LinearSVM
 `decision_function()` of LinearSVM returns the distance between an input feature and its hyperplane. The `find_car` function considers positive prediction with short distance to hyperplane (< 1.0) as false.  


### Bounding Positive Detections (Step 3-3 in `Vehicle_Detection.ipynb`)   
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  


#### Here is the output of my pipeline with their corresponding heatmap and labeled map :  

![alt text][output]
![alt text][label_map]

---

## Video Implementation

### Averaging last six `heatmap`s (Step 3-3 in `Vehicle_Detection.ipynb`)   
 `find_car` function stores the positions of positive detections - `heatmap` in each frame of the video in the buffer having the depth of 6. By using the buffer, I could take the mean of last six `heatmap`s and remove 

Here's a [link to my video result](./result_project_video.mp4) and also available in [Youtube](https://youtu.be/vX57KumVp8k)

---

## Discussion

### To perfectly remove **false detection**.
I guess the following approaches can make the pipeline more robust.  
1) Gathering more labeled dataset from the false-detection cases
  Additional training dataset extracted from false-detection cases in the project video could increase the accuracy of classifier in this project.
2) Trying to find optimal parameters for the classifier  
  I've just tunned the parameters for feature extraction. I think that the accuracy of the classifier can be improved by some efforts on tunning the parameters of classifier
