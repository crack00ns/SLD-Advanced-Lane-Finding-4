# Advanced Lane Finding
Implementation of  Project 4 of Self Driving Car **: Advanced Lane Finding**.
---

**Advanced Lane Finding Project**
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/corners_found_calibration2.jpg "Found Corners"
[image2]: ./output_images/test_cal_undist.png "Distortion Correction 1"
[image3]: ./output_images/test_undist.png "Distortion Correction 2"
[image4]: ./output_images/pipeline.png "Binary Threshold Pipeline"
[image5]: ./output_images/final_masked_img.jpg "Binary Thresholding and Masking"
[image6]: ./output_images/binary_threshold.png "Binary Threshold Example 1"
[image7]: ./output_images/binary_threshold2.png "Binary Threshold Example 2"
[image8]: ./output_images/pt1.png "Source and destination points for Perspective Transform"
[image9]: ./output_images/pt2.png "Pespective Transform on Binary Images"
[image10]: ./output_images/histogram.png "Histogram"
[image11]: ./output_images/detected_lanes.png "Detected Lanes with Sliding Window"
[image12]: ./output_images/no_sliding_window.png "Detected Lanes with previous lane positions"
[image13]: ./output_images/plot_lanes.jpg "Final Image with Marked Lanes"

---
## Project code
All main part of code for this project is implemented in the following `./laneutils.py` file. The IPython notebook `./advanced_lanes.ipynb` has a streamlined code showing how to run the code from laneutils.py on a sample image and video files. Diagonostics images can be obtained by setting `debug` parameter in LaneDetector object to be true. All the generated images are stored in `output_images` folder.


### 1. Camera Calibration
The code for this step is contained in lines 47-124 of `./laneutils.py` within class `CalibrateCamPerspectiveTransform`. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the results as shown below. All the required code is part of `CalibrateCamPerspectiveTransform.caliberate()` and `CalibrateCamPerspectiveTransform.undistort()` functions.

![Found Corners][image1]
*Detecting Checkerboard Corners in an Image*

![Distortion Correction][image2]
*Distortion Correction on Checkerboard Image*

![Distortion Correction][image3]
*Detecting Checkerboard on Lane Image*
---

### 2. Gradient Threshold
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 383-492 in `./laneutils.py`). I tried various combinations and thesholds and after several hit and trials, I settled with a sobel kernel size of 3 with following Thresholds to create each binary thresholded image: 

| Item        | Thresholds   | 
|:-------------:|:-------------:| 
| SobelX Thresholds    | (50 , 200) | 
| SobelY Thresholds    | (50 , 200) | 
| Sobel Magnitude Thresholds    | (50 , 200) | 
|Sobel Direction Thresholds    | (0.3, 1.3) | 
| H Thresholds in HLS Space    | (20,  100) | 
| L Thresholds in HLS Space    | (200, 255) | 
| S Thresholds in HLS Space    | (170, 255) | 

Finally I combined the binary with the following rules so that lanes can be seen well
`combined[((gradx == 1)&(grady ==1))|((mag_binary == 1) & (dir_binary == 1))|(l_binary == 1) |(s_binary == 1)]=1`
Here is pipeline of the whole thresholding process:
![alt text][image4]
*Binary Thresholding Pipeline*

Thereafter I applied a mask to remove the remaining artefacts from the road so that only lanes are visible. Please check `region_of_interest()` function in `.\laneutils.py` on details how masking was performed. Here is the final output:
![alt text][image5]
*Applied Region of Mask*

Here are some samples of images side by side (original image and thresholded image after distortion correction)
![alt text][image6]
![alt text][image7]
---

### 3. Perspective Transform
The code for my perspective transform and inverse transform includes a function called `warp()` and `unwarp()` which is part of `CalibrateCamPerspectiveTransform` class in `./laneutils.py`. These functions take input an image (`img`) and outputs the warped or unwarped images. The transformation matrix is obtained in `CalibrateCamPerspectiveTransform` constructor using hardcoded source (`src`) and destination (`dst`) points. A series of combinations of source and destination points were tested to makes the lanes appear parallel. This experimentation was done in codecell [6] of ipython notebook file `./advanced_lanes.ipynb`. Here is the final selection of source and destination points applied on an image with straight lanes assuming the road is flat.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 208, 720      | 350, 720        | 
| 580, 460      | 350, 0      |
| 705, 460     | 900, 0      |
| 1120, 720      | 900, 720        |

![alt text][image8]
*Image used to map source and destination points shows that lanes look parallel after transform with the choise of src/dst points*

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart after binary thresholding and masking to verify that the lines appear parallel in the warped binary image as shown below.
![alt text][image9]
---

### 4. Identifying Lane Pixels and Masking the Lane
To identify lane pixels, I used a technique discussed in lectures where first I take a histogram of lower half of the image. This is implemented in `.\laneutils.py` in `LaneDetector` class as part of `peak_detection(binary_warped)` function in lines 178-194. The output is shown below.
![alt text][image10]

The peaks of the histogram correspond to the lane positions which were used to mark initial left and right positions and detect lane pixels using a sliding window approach was used as discussed in lectures. Then a second degree polynomial was fitted for the lane positions. Lanes lines are tracked with a `Line` class that keeps track of a bunch of parameters such as best fit polynomial (averaged over 5 frames), current fit polynomial and current x and y points for each lane line. Here is the output of sliding window procedure:
![alt text][image11]

If the lanes were detected in previous frame, we used targeted search in region (with a margin of 100) on previously detected lanes as shown below.
![alt text][image12]

The entire code of this procedure is part of `detect_and_plot_lanes()` function of the `LaneDetector` class. If the lanes were detected, lane curvature is calculated with `calculate_curvature()` function given in lines 521-530 in `./laneutils.py`. Sanity checks were performed on lanes whether they are parallel and separated by the right distance. Vehicle position is also calculated based on how much center of the lane is offset by the center of the image. Finally detected lane region was masked with green color and unwarped and added to the original along with curvature and vehicle position as shown below: 
![alt text][image13]

---

### Pipeline (video)
The same pipeline is applied to videos. The pipeline is implemented in `LaneDetector`'s `find_lane_img_frame` function. Here's a [link to my video result](./project_video_with_marked_lanes.mp4) or [click here ](https://www.youtube.com/watch?v=tHk7W1KPjhk "Advanced Lane Finding") to watch it on youtube

---

### Discussion and Issues Faced
I think the project requires more than 10 hours as it is quite a challenging project. 
1. **Binary Thresholding Pipeline**: The binary threshold pipeline is currenty not very robust for videos/images under very different lighting conditions. So there is definetely room for improvement. More color spaces such as YUV and better selection of thresholds need to be explored to see if lanes can be masked more effectively. This could be useful for the challenge video.
2. **Lane Finding Approach**: Currently, the sliding window approach assumes that both lanes can be seen effectively but sometimes while switching lanes this could be a problem. Also on a hilly terrain, perspective transform may not work well and if the lanes are very zig-zaggy then the sliding window along with 2nd order polynomial filling may not work especially applicable for the harder challenge video.   