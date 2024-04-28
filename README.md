## Image Stitching Comparison README

### Overview
This repository contains scripts for comparing four different methods for image stitching: SIFT, Harris Corner Detection, FAST, and CNN models. Each method is implemented separately, and after obtaining the stitched image, the Mean Squared Error (MSE) and Structural Similarity Index (SSI) are computed for comparison.

### Dependencies
- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- SciPy

### Setup
1. Ensure Python and the required libraries are installed.
2. Clone or download the repository.
3. Place the images to be stitched in a designated folder.
4. Update the `folder_path` variable in each script to point to the folder containing the images.

### Usage
1. Run each script separately for the desired method:
   - For SIFT: `python SIFT_Based_Approach.py`
   - For Harris Corner Detection: `python HarrisCorner.py`
   - For FAST: `python FAST.py`
   - For CNN models: `python CNN_Based_Approch.py`
2. Each script will process the images using the specified method.
3. After stitching, the scripts will compute the MSE and SSI for each method.
4. The results will be displayed or saved as specified in the scripts.

### Notes
- Each method is implemented separately within its corresponding script.
- Parameters such as thresholds or model architectures can be adjusted within each script.
- Ensure that images are properly preprocessed and formatted according to the requirements of each method.
- Results may vary based on image quality, content, and other factors.
- It's recommended to experiment with different methods and parameters for optimal results.

### References
- OpenCV documentation: [https://docs.opencv.org](https://docs.opencv.org)
- NumPy documentation: [https://numpy.org/doc/](https://numpy.org/doc/)
- Matplotlib documentation: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
- SciPy documentation: [https://docs.scipy.org/doc/](https://docs.scipy.org/doc/)


