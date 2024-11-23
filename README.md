
  
# OpenCV Car Detection    
#### Made by [Kubiczeek](https://github.com/Kubiczeek)       
This script uses OpenCV to detect cars in an image.
It uses the YOLOv11 model for the detection.
It saves the results in a new image with the detected cars outlined.
 
---
### Requirements    
- Python 3    
- OpenCV  
- Ultralytics  
---
### Running the script    
The script has a few optional arguments that can be passed when running it:    
`python main.py --image PATH_TO_IMAGE [--conf CONFIDENCE_THRESHOLD] [--dest DESTINATION_DIRECTORY]` - `--image PATH_TO_IMAGE` - The path to the image that you want to detect cars in.  
- `--conf CONFIDENCE_THRESHOLD` - The confidence threshold for the detection. The default value is 0.3.    
- `--dest DESTINATION_DIRECTORY` - The directory where the output image will be saved. The default value is the same directory as original.     
    
#### Example usage:  
* `python main.py --image images/car.jpg --conf 0.5 --dest output/` - This will detect cars in the `car.jpg` image with a confidence threshold of 0.5 and save the output image in the `output/` directory.  
* `python main.py --image images/car.jpg` - This will detect cars in the `car.jpg` image with the default confidence threshold of 0.3 and save the output image in the same directory as the original image.  
* `python main.py --image images/car.jpg --dest output/` - This will detect cars in the `car.jpg` image with the default confidence threshold of 0.3 and save the output image in the `output/` directory.  
* `python main.py --image images/car.jpg --conf 0.5` - This will detect cars in the `car.jpg` image with a confidence threshold of 0.5 and save the output image in the same directory as the original image.