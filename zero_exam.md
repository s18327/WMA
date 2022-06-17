# Test 1

1. Edge detector Canny:
- [x] use double thresholding.
- [ ] uses a single Sobel filter instead of Gaussian filter.
- [ ] uses a set of Sobel filters instead of a Gaussian filter.

2. Linear edge filters can:
- [ ] detect only vertical lines.
- [x] detect skewed lines.
- [ ] detect only horizontal straight lines.

3. Laplace operator: 
- [x]  detects edges taking into account the second derivative. 
- [ ] is resistant to noise.
- [ ] is dependant on the edge orientation.

4. Computer vision:
- [x] is a scientific discipline dealing with systems which extract information from images.
- [ ] deals with generating images with the use of computers.
- [ ] is an interactive system in which the objects which are located in real world are amplified by computer-generated perceptual information.

5. Points lying on the edge you can detect:
- [ ] detecting the local minima of the second derivative.
- [x] detecting zero crossing of the second derivative. 
- [ ] using morphological operations.

6. Lucas - Kanade optical flow:
- [ ] detects the actual movement between two images.
- [x] uses local techniques.
- [ ] is used to detect large, significant changes.
  
7. Harris Corner Detector:
- [ ] measures change in intensity moving a window in each of the eight major directions.
- [x] is invariant for image rotation.
- [ ] is invariant for scaling.

8. Convolutional neural network: ???
- [x] thanks to the Convolutional layer, it extracts the characteristic features of the image.
- [ ] it must not use any connecting layers between the Convolutional layers.
- [ ] it does not allow the use of RGB - encoded color images.
  
9.  So that the line filter does not change the energy of the image:
- [ ] Brightness distribution of pixels should have a zero mean value.
- [x] The sum of the filter weights should be one.
- [ ] The sum of the filter weights should be zero.

10. In line filters:
- [ ] new value of the point is calculated based on the morphological operator.
- [x] neighborhood of a point is used to compute a new point value.
- [ ] the global image model is used.

11. Implemented in OpenCV Brute-Force Matcher:
- [ ] It uses the distance function as the fit criterion 
- [ ] Attempts to match the feature in the first set only with the predefined range of features in the second.
- [ ] Takes the feature descriptor of the second set and tries to match all the feature points of the first set.

12. First derivative.
- [x] gives thick edges.
- [ ] they give a double edge.
- [ ] significantly improve the fine details in relation to the second derivative.

13. Convolutional neural network:
- [x] It includes at least one convolutional layer.
- [ ] It cannot be used to classify images.
- [ ] It includes in its structure convolutional areas which, by definition, do not take part in the analysis

14. Closing operation can be presented as:
- [ ] D(E(x))
- [x] E(D(x))
- [ ] E(x) * D(x)

15. Visual Odometry:
- [ ] Works in every environment.
- [ ] Requires at least two images to determine the robot's displacement,
- [ ] Requires the use of the Optical Flow technique to determine position change.

---  
# Test version 2

1. Sift algorithm:
- [x] It is based on the difference between the levels of the sub-octave Gaussian pyramid.
- [ ] It is based on the difference between the layer of the sub-hexadecimal Gaussian pyramid.
- [ ] Does not use Gauss filters in the algorithm.

2. Convolutional neural network. 
- [ ] Effectively, it can only be used to classify images.
- [x] Architecturally, similar to the visual cortex, it has local perceptual fields.
- [ ] It does not allow the use of RGB-encoded color images.

3. For algorithms based on feature matching.
- [ ] The matching strategy / algorithm may have an impact on the number and quality of fitted fiducials.
- [ ] The matching strategy / algorithm cannot affect the number and quality of matching characteristic points because in this way different strategies would give different results.
- [ ] The matching strategy / algorithm can only affect the quality of the matching of the characteristic points.

4. Linear edge filters can: 
- [ ] Detect only vertical lines.
- [x] Detect skew lines.
- [ ] Detect only horizontal straight lines.

5. BRIEF (Binary Robust Independent Elementary Features):
- [ ] binary series are used for feature matching using Hamming metrics.
- [ ] uses simplified floating point notation.
- [x] makes it easy to find descriptors.

6. Implemented in OpenCV FLANN based Matcher:
- [x] It is dedicated to quickly find the nearest neighbor in large data sets.
- [ ] It prioritizes the accuracy of the match over speed of the search.
- [ ] it cannot be applied to descriptors defined by the SIFT algorithm.

7. Line filtration: 
- [ ] Always  maintains the size of the image.
- [x] Can keep the image size by applying the appropriate padding.
- [ ] Always keeps the image size for a 2x2 kernel only.

8. Edge detector Canny:
- [x] use double thresholding.
- [ ] uses a single Sobel filter instead of Gaussian filter.
- [ ] uses a set of Sobel filters instead of a Gaussian filter.

9. The actual characteristic feature of the image:
- [x] it should be resistant to additional  noise in the image.
- [ ] it can't be local,
- [ ] it only occurs as a given image scale.

10. Harris Corner Detector:
- [ ] measures change in intensity moving a window in each of the eight major directions.
- [x] is invariant for image rotation.
- [ ] is invariant for scaling.

11. Characteristic points according to the SIFT algorithm are defined by:
- [ ] Just the location.
- [x] Set: position, scale, orientation.
- [ ] Set: position, scale, orientation, intensity.

12. MaxPooling Operation:
- [ ] It cannot be used as the connecting layer of the convolutional neural network.
- [x] It is analogous to the image scaling operation.
- [ ] For the determined kernel, it chooses the lowest value in it as a result.

13. Discrete wavelet transform:
- [ ] It is a Fourier frequency analysis.
- [x] It is always implemented as a filter bank.
- [ ] It does not depend on the scale and on time?
  
14. Convolutional neural network:
- [x] It includes at least one convolutional layer.
- [ ] It cannot be used to classify images.
- [ ] It includes in its structure convolutional areas which, by definition, do not take part in the analysis.

15. To blur the image you can apply:
- [x] box filter.
- [ ] Sobel filter.
- [ ] the Laplacian.