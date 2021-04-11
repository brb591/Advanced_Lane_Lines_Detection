import cv2
import os
import glob
import argparse
from camera_calibrator import CameraCalibrator
from image_processor import ImageProcessor

#
# Undistort a single image
#
#
def handle_image(file_name, output_dir, calibration_matrix, calibration_distortion):
  # Get the image
  image = cv2.imread(file_name)

  # Create an instance of the ImageProcessor
  image_processor = ImageProcessor(calibration_matrix, calibration_distortion)

  # Process the image
  processed_image = image_processor.undistort(image)
  
  if output_dir is None:
    # Show both the original and the processed images
    cv2.imshow('Original', image)
    cv2.imshow('Processed', processed_image)
    
    print('Press any key to dismiss')
    # Wait for the user to press any key
    cv2.waitKey(0)
  else:
    # Save the image to the output_dir
    output_file = os.path.join(output_dir, 'calibration_' + os.path.basename(file_name))
    print('Saving undistorted image to %s'%output_file)
    cv2.imwrite(output_file, processed_image)
    print('Done.')


#
# main function
#
#
#

# Calibration Source
calibration_source = './camera_cal/calibration*.jpg'

# Create the parser
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', nargs='+', type=str, help='The input file(s)')
parser.add_argument('-o', '--output', nargs='?', type=str, help='The output directory')

# Get the arguments
args = parser.parse_args()

# Get the calibration information
print("Calibrating the camera...")
camera_calibrator = CameraCalibrator(calibration_source)
matrix, distortion = camera_calibrator.calibrate()
print("Done.")

# Process the test image(s)
for input_file in args.input:
  handle_image(input_file, args.output, matrix, distortion)


# Closes all the frames
cv2.destroyAllWindows()