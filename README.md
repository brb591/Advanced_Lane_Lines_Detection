# Advanced Lane Detection
## DGMD E-17 Assignment 5 - Brian Bauer

I have created two python programs, ```calibration_test.py``` and ```lane_detection.py``` which combine to satisfy the requirments of Assignment 5.  Both of these programs use two classes that I have created (```CameraCalibrator``` and ```ImageProcessor```) to accomplish their goals.

### CameraCalibrator (camera_calibrator.py)
The ```CameraCalibrator``` class has only one purpose: to conduct camera calibration when provided with a large number of chessboard images. When creating an instance of the class, you must provide a directory that contains the images.  To calibrate the camera, invoke the ```calibrate``` method.  When finished, this method returns the calibration matrix and distortion for the camera.  These values can be used to undistort images using the ```ImageProcessor``` class.

#### ```calibrate(self)```
This method gets all of the images in the directory that was provided at class instantiation.  The cv2 ```findChessboardCorners``` method is used.  Once all corners are found in all images, the cv2 ```calibrateCamera``` method is invoked using this data.  The calibration matrix and distortion are found and returned to the callling method.

### ImageProcessor (image_processor.py)
The ```ImageProcessor``` class can be used to detect lane lines from a single image.  First, create an instance of the class while providing the calibration matrix and distortion.  The intended use case is to then invoke the ```process_image``` method for each image you wish to process.  When successful, the returned image contains the original image with the lane superimposed on it as well as the radius of the lane curve and the distance from the center of the lane.  If you are processing a video file, invoke this method repeatedly, once per video frame.

#### ```process_image(self, image)```
This is the main method to be used for processing images.  The workflow for this method is as follows:
- Undistort the image
- Do a perspective transform to get a "bird's eye view" of the lane ahead
- Get a grayscale version of the transformed image
- Get the S Channel (in the HLS space) of the transformed image
- Generate a binary image from the grayscale, using an absolute value Sobel threshold (in the X orientation)
- Generate a binary image from the S Channel, using a threshold
- Combine the two binary images into a single binary image
- Use a sliding window search to find left and right lane lines
- Draw the lane lines on a copy of the undistorted (but not perspective transformed) image
- Calculate the radius of the lane curvature, as well as the distance from the centerline
- Put the radius and distance text onto the annotated image
- Return the image
#### ```undistort(self, image)```
This method uses the cv2 library to undistort the provided image, using the calibration maxtrix and distortion from the camera calibration process (which was provided when the class was instantiated).
#### ```hls_binary(self, image, thresh)```
This method creates a black image of the same size as the provided image, and sets each pixed to 1 if the equivalent pixel in the provided image has a value within the threadholding range.  The resulting image is then returned to the calling method.
#### ```abs_sobel_thresh(self, image, orientation, thresh)```
This method applies the Sobel transformation along the specified orientation.  This absolute value of this result is generated, and scaled to be from 0 to 255.  A black image of the same size as the provided image is created, with each pixed set to 1 if the scaled pixed value lies within the thresholding range.  The resulting image is then returned to the calling method.
#### ```image_histogram(self, wrapped_image)```
This method calculates the histogram for the provided image.  A histogram shows the distribution of data by summing the data in each "bin" in the image.  The histogram is returned to the calling method.
#### ```sliding_window_search(self, wrapped_image)```
This method finds the lane lines in the provided image.  The image is divided into horizontal windows and processing begins at the bottom of the image.  This represents the area of the road that is closest to the car.  An approximate area in which the left and right lanes are expected to be found is set, and a margin on either side of these two areas is set.

The boxes created within that window are analyzed and as long as the total number of pixels exceeds a minimum count, then the mean X-axis values for the pixels are determined (one for the left lane and one for the right lane).  These two means are used as the center of the left and right lane finding boxes in the next horizontal window.

Once all horizontal windows haven been analyzed, we fit a polyline across all mean values for the left lane and another for the right lane.  This becomes our left and right lane line.  If the number of good fits exceeds a threshold, the good points are used the next time the sliding window search is invoked.  This makes the sliding window search a little bit more efficient when processing a video stream.
#### ```draw_unwrapped(self, original_image, wrapped_image, left_fit, right_fit)```
This method draws lane lines using the "birds eye view perspective, and then warps the overlay back to the original perspective and adds it onto the top of the original image.  The image that is returned contains the original image and the lane lines.
#### ```calculate_curve_radius(self, wrapped_image, left_fit, right_fit)```
Using the "birds eye view" perspective and an assumed number of pixels per meter in the X and Y dimension, we determine the radius of curvature as defined by [Wikipedia](https://en.wikipedia.org/wiki/Radius_of_curvature#In_2D).  This is done for both the left and right lanes and averaged.

When calculating the distance from center, we set the center of the car as the center of the image.  We then determine the difference between the center of the image and the center of the two lane lines at the bottom of the image (in pixels).  This is then converted to meters using our pixels per meter assumption.
#### ```convert_gray(self, image, read_style)```
This method returns a grayscale version of the provided image.  How it treats the provided image is different based upon whether the image was loaded with cv2 or with matplotlib.
#### ```convert_hls(self, image, channel, read_style)```
This method returns the requested channel after first converting the image to hls colorspace.  How it treats the provided image is different based upon whether the image was loaded with cv2 or with matplotlib.
#### ```perspective_transform(self, image)```
The perspective transformation starts with an undistorted image.  We choose four points on this image that correspond to a rectangle in the transformed image.  The source and destination points are used by cv2 to create a new image that has been transformed to a new perspective.

## Usage
### calibration_test.py
Test the camera calibration.  The output is either displayed on screen or saved to a file.  If saved to a file, ```calibration_``` is appended to the beginning of the file name to indicate that it is calibration output.

The output file is saved in the test_images directory as calibration_test2.jpg:
```
python3 calibration_test.py -i test_images/test2.jpg -o output_images
```
If no output directory is provided, the processed image is shown on the screen:
```
python3 calibration_test.py -i test_images/test2.jpg
```
Multiple test images can be specified:
```
python3 calibration_test.py -i 'test_images/test2.jpg' 'test_images/test3.jpg' 'test_images/test4.jpg' -o output_images
```

### lane_detection.py
Detect lane lines on images or videos.  The output is displayed on a screen.  Saving file output is not supported.
```
python3 lane_detection.py -i project_video.mp4
```

## Analysis
### What can be improved
- The perspective transform is not perfect, and it ends up throwing away much of the frame.  If the lane lines are outside of this area, they will never be found.
- The channels and thresholds are hardcoded.  If these values were parameterized, then we could have a system whereby if a few frames have difficulty identifying lane lines, other approaches could be tried and if they are successful then they would be used for future frames in the video.
### What can cause the algorithm to fail
- Lane lines that are worn and faded.
- Seams in the pavement that run parallel to lane lines (to be fair, this messes up the lane tracking in my car).
- When the distance between the lanes is not within the range of expectations, the perspective transformation and the window search functions can fail.