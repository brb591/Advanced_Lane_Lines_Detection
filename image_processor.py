import numpy as np
import matplotlib.pyplot as plt
import cv2

class ImageProcessor:
    def __init__(self, calibration_matrix, calibration_distortion):
        self.calibration_matrix = calibration_matrix
        self.calibration_distortion = calibration_distortion
        self.base_hyperparameter = [0,0]
        self.high_confidence_counter = 0
        self.total_frame_counter = 0
    
    def process_image(self, image):
        self.total_frame_counter += 1

        # Undistort the image
        undistorted_image = self.undistort(image)

        # Do a perspective transform
        warped, unwarped = self.perspective_transform(undistorted_image)

        # Get Grayscale
        gray = self.convert_gray(warped, "cv2")

        # Get S-Channel
        s_channel = self.convert_hls(warped, "s", "cv2")

        # Get the binary images for both gradient and hls channel
        sxbinary = self.abs_sobel_thresh(gray, 'x', (20,100))
        s_binary = self.hls_binary(s_channel, (170,255))

        # Only useful for debugging
        # Stack the binary images
        #color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        # Combine the binary images
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        # Find the lanes using a sliding window search
        sliding_windows = self.sliding_window_search(combined_binary)

        # Draw the lanes on the unwarped image
        left_fit = sliding_windows[0]
        right_fit = sliding_windows[1]
        final_image = self.draw_unwrapped(image, warped, unwarped, left_fit, right_fit)

        # Calculate the curvature of the road and the distance from center 
        radius, distance = self.calculate_curve_radius(warped, left_fit, right_fit)
        cv2.putText(final_image, "Radius of curve is " + str(int(radius))+ "m", (100,100), 2, 1, (255,255,0), 2)
        cv2.putText(final_image,"Distance from center is {:2f}".format(distance)+ "m", (100,150), 2, 1, (255,255,0),2)

        return final_image

    def undistort(self, image):
        return cv2.undistort(image, self.calibration_matrix, self.calibration_distortion, None, self.calibration_matrix)

    def hls_binary(self, image, thresh):
        binary_image = np.zeros_like(image)
        binary_image[(image > thresh[0]) & (image <= thresh[1])] = 1
        
        return binary_image

    def abs_sobel_thresh(self, image, orientation, thresh):
        if orientation == 'x':
            abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0))
        else:
            abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1))
        
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary_output

    def image_histogram(self, wrapped_image):
        return np.sum(wrapped_image[wrapped_image.shape[0]//2:,:], axis=0)

    def sliding_window_search(self, wrapped_image):
        out_img = np.dstack((wrapped_image, wrapped_image, wrapped_image)) * 255

        # If the prior image was processed and had a great sliding window
        #   search, use that base as a starting point.  Otherwise, calculate
        #   a starting point based upon a histogram
        if self.base_hyperparameter[0] > 0 and self.base_hyperparameter[1] > 0:
            # High confidence on lane line location
            leftx_base = self.base_hyperparameter[0]
            rightx_base = self.base_hyperparameter[1]
            self.high_confidence_counter += 1
            print("High confidence frame %d of %d"%(self.high_confidence_counter,self.total_frame_counter))
        else:
            # Low confidence on lane line location
            histo = self.image_histogram(wrapped_image)
            midpoint = np.int(histo.shape[0] / 2)
            leftx_base = np.argmax(histo[:midpoint])
            rightx_base = np.argmax(histo[midpoint:]) + midpoint

        num_windows = 9
        window_height = np.int(wrapped_image.shape[0] / num_windows)
        nonzero = wrapped_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        margin = 100
        minpix = 50
        left_lane_indexes = []
        right_lane_indexes = []

        good_count = 0
        window_count = 0
        next_base_left = 0
        next_base_right = 0
        
        for window in range(num_windows):
            # Identify window boundaries in x and y (and right and left)
            top_y = wrapped_image.shape[0] - (window + 1) * window_height
            bottom_y = wrapped_image.shape[0] - window * window_height
            left_lane_leftx = leftx_current - margin
            left_lane_rightx = leftx_current + margin
            right_lane_leftx = rightx_current - margin
            right_lane_rightx = rightx_current + margin
            
            cv2.rectangle(out_img, (left_lane_leftx, top_y), (left_lane_rightx, bottom_y), (0, 255, 0), 2) 
            cv2.rectangle(out_img, (right_lane_leftx, top_y), (right_lane_rightx, bottom_y), (0, 255, 0), 2) 

            # Identify the nonzero pixels in x and y within the window
            good_left_indexes = ((nonzeroy >= top_y) & (nonzeroy < bottom_y) & 
            (nonzerox >= left_lane_leftx) &  (nonzerox < left_lane_rightx)).nonzero()[0]
            good_right_indexes = ((nonzeroy >= top_y) & (nonzeroy < bottom_y) & 
            (nonzerox >= right_lane_leftx) &  (nonzerox < right_lane_rightx)).nonzero()[0]

            left_lane_indexes.append(good_left_indexes)
            right_lane_indexes.append(good_right_indexes)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_indexes) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_indexes]))
                good_count += 1
            if len(good_right_indexes) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_indexes]))
                good_count += 1
            
            # Keep track of the center of the very first windows for potential future use
            if window_count == 0:
                next_base_left = leftx_current
                next_base_right = rightx_current
            window_count += 1
        
        # Concatenate the arrays of indices
        left_lane_indexes = np.concatenate(left_lane_indexes)
        right_lane_indexes = np.concatenate(right_lane_indexes)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_indexes]
        lefty = nonzeroy[left_lane_indexes] 
        rightx = nonzerox[right_lane_indexes]
        righty = nonzeroy[right_lane_indexes] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # If we have a high confidence that we've found 
        if good_count == num_windows * 2:
            self.base_hyperparameter[0] = next_base_left
            self.base_hyperparameter[1] = next_base_right
        else:
            self.base_hyperparameter[0] = 0
            self.base_hyperparameter[1] = 0

        return left_fit, right_fit

    def draw_unwrapped(self, original_image, wrapped_image, unwrapped_image, left_fit, right_fit):
        height, width, _ = wrapped_image.shape
        ploty = np.linspace(0, wrapped_image.shape[0]-1, wrapped_image.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty +left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 
        warp_zero = np.zeros_like(wrapped_image).astype(np.uint8)
        color_warp = np.zeros_like(wrapped_image).astype(np.uint8)
    
        ploty = np.linspace(0, height - 1, num=height) # to cover same y-range as image
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        points = np.int_([pts])

        cv2.fillPoly(color_warp, points, (0, 255, 0, 0))
        cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
        cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

        # Warp the blank back to original image space using inverse perspective matrix (unwrapped_image)
        newwarp = cv2.warpPerspective(color_warp, unwrapped_image, (width, height)) 

        result = cv2.addWeighted(original_image, 1, newwarp, 0.5, 0)
        return result

    def calculate_curve_radius(self, wrapped_image, left_fit, right_fit):
        ym_per_pix = 30 / wrapped_image.shape[0] * 0.625 # meters per pixel in y dimension
        xm_per_pix = 3.7 / wrapped_image.shape[1] * 0.7 # meters per pixel in x dimension
        
        ploty = np.linspace(0, wrapped_image.shape[0]-1, wrapped_image.shape[0] )
        leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty +left_fit[2]
        rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        car_position = wrapped_image.shape[1] / 2

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        
        max_y = np.max(ploty)

        # _fit is a 2 degree polynomial of the form y = ax^2 + bx + c
        # From wiki: https://en.wikipedia.org/wiki/Radius_of_curvature#In_2D
        left_first_deriv = 2 * left_fit_cr[0] + left_fit_cr[1]
        left_secnd_deriv = 2 * left_fit_cr[0]
        left_radius = int(((1 + (left_first_deriv ** 2) ** 1.5) / np.absolute(left_secnd_deriv)))
        
        right_first_deriv = 2 * right_fit_cr[0] + right_fit_cr[1]
        right_secnd_deriv = 2 * right_fit[0]
        right_radius = int(((1 + (right_first_deriv ** 2) ** 1.5) / np.absolute(right_secnd_deriv)))
        
        left_lane_bottom = (left_fit[0] * max_y)**2 + left_fit[1] * max_y + left_fit[2]
        right_lane_bottom = (right_fit[0] * max_y)**2 + right_fit[1] * max_y + right_fit[2]
        
        actual_position= (left_lane_bottom + right_lane_bottom) / 2
        
        distance = np.absolute((car_position - actual_position) * xm_per_pix)
        
        # Now our radius of curvature is in meters
        #print(left_curverad, 'm', right_curverad, 'm')
        return (left_radius + right_radius) / 2, distance
        # Example values: 632.1 m    626.2 m
    
    def convert_gray(self, image, read_style):
        if read_style == "cv2":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def convert_hls(self, image, channel, read_style):
        if channel == 'h':
            cn = 0
        elif channel == 'l':
            cn = 1
        else:
            cn = 2
        
        if read_style == "cv2":
            ct = cv2.COLOR_BGR2HLS
        else:
            ct = cv2.COLOR_BGR2HLS

        hls = cv2.cvtColor(image, ct)
        return  hls[:,:,cn]

    def perspective_transform(self, image):
        imshape = image.shape
        height = imshape[0]
        width = imshape[1]
        width_ratio = 0.457
        height_ratio = 0.625
        top_left = (width * width_ratio, height * height_ratio)
        top_right = (width * (1 - width_ratio), height * height_ratio)
        bottom_left = (100, height)
        bottom_right = (width - 100, height)

        src_vertices = np.float32([top_left, top_right, bottom_left, bottom_right])
        d1 = [100, 0]
        d2 = [width - 100, 0]
        d3 = [100, height]
        d4 = [width - 100, height]
        dst_vertices = np.float32([d1, d2, d3, d4])
        matrix = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
        warped = cv2.warpPerspective(image, matrix, (width, height))
        # We also calculate the oposite transform as we'll need it later
        unwarped = cv2.getPerspectiveTransform(dst_vertices, src_vertices)
        # Return the resulting image and matrix 
        return warped, unwarped