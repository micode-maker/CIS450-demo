# CIS450-demo

## Introduction
This repo illustrates best practice README file generation.

## Projects
Open-CV image processing demos.

### Edge Detection and Image Blending
Edge detection identifies areas in an image where pixel intensity changes sharply, which usually correspond to object boundaries. The process begins by converting the image to grayscale and applying a Gaussian blur to reduce noise. Sobel operators are then used to compute horizontal and vertical gradients. These gradients are combined to calculate the gradient magnitude, which represents edge strength. A binary threshold is applied so that only the strongest edges are retained, reducing spurious noise.

Once edges are detected, they are converted back into a color image so they can be combined with the original. Image blending is done using a weighted sum of the original image and the edge image. By adjusting the blending weights, the edges can be emphasized without overpowering the original image. This allows important structural details to stand out while preserving the overall appearance of the image.

## Resources
<img src="Images/OpenCV_logo_black_.png" alt="OpenCV, logo with black text" width="100"/>

[Open-CV](https://opencv.org/)
