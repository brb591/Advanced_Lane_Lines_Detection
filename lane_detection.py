import cv2
import os
import glob
import argparse
from camera_calibrator import CameraCalibrator
from image_processor import ImageProcessor

#
# handle_image runs the pipeline on a single, undistorted image
#
#
def handle_image(fileName, output_dir, calibration_matrix, calibration_distortion):
  # Get the image
  image = cv2.imread(fileName)

  # Create an instance of the ImageProcessor
  image_processor = ImageProcessor(calibration_matrix, calibration_distortion)

  # Process the image
  processed_image = image_processor.process_image(image)
  
  if output_dir is None:
    # Show both the original and the processed images
    cv2.imshow('Original', image)
    cv2.imshow('Processed', processed_image)
    
    print('Press any key to dismiss')
    # Wait for the user to press any key
    cv2.waitKey(0)
  else:
    # Save the image to the output_dir
    output_file = os.path.join(output_dir, os.path.basename(fileName))
    print('Saving image to %s'%output_file)
    cv2.imwrite(output_file, processed_image)
    print('Done.')

#
# handle_video runs the pipeline on each undistorted frame of a video file
#
#
def handle_video(fileName, calibration_matrix, calibration_distortion):
  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  cap = cv2.VideoCapture(fileName)

  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")

  # Create an instance of the ImageProcessor
  image_processor = ImageProcessor(calibration_matrix, calibration_distortion)

  print('Press q to dismiss or wait for the video to end')
  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
      #Process the frame
      processed_frame = image_processor.process_image(frame)
      
      # Display the original frame in its own window
      cv2.imshow('Original',frame)
      # Display the processed frame in a window
      cv2.imshow('Processed',processed_frame)
      # Press Q on keyboard to  exit
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Break the loop
    else: 
      break

  # When everything done, release the video capture object
  cap.release()


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
  # Determine the file type
  filename, extension = os.path.splitext(input_file)

  if extension == ".jpg":
    print("Processing Image " + input_file)
    handle_image(input_file, args.output, matrix, distortion)
  elif extension == ".png":
    print("Processing Image " + input_file)
    handle_image(input_file, args.output, matrix, distortion)
  elif extension == ".mp4":
    print("Processing Video " + input_file)
    handle_video(input_file, matrix, distortion)
  else:
    print("Unknown extension: " + extension)


# Closes all the frames
cv2.destroyAllWindows()