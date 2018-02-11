# SDCND-P5
The purpose of this repo is to hold all files necessary for project 5 (Vehicle Detection) of Udacity's self driving car nanodegree program.

## Vehicle detection project

**Project goals**
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Contents of this repo
1. generated - Folder that contains some images used in documentation. The generated project video is also available in this folder.
2. findCars.py - The main python file that runs the vehicle detection algorithm.
3. searchAndClassify.py - The python file which trains and tests the classifier based on selected features from the test set.
3. lesson_functions.py - This python file contains all the necessary functions used by other python scripts.
4. writeup.md - A report for the project.
