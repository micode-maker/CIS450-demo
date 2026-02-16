# CIS450-demo

## Introduction
This repo illustrates best practice README file generation.

## Projects
Open-CV image processing demos.

## Edge Detection and Image Blending
Edge detection identifies areas in an image where pixel intensity changes sharply, which usually correspond to object boundaries. The process begins by converting the image to grayscale and applying a Gaussian blur to reduce noise. Sobel operators are then used to compute horizontal and vertical gradients. These gradients are combined to calculate the gradient magnitude, which represents edge strength. A binary threshold is applied so that only the strongest edges are retained, reducing spurious noise.

Once edges are detected, they are converted back into a color image so they can be combined with the original. Image blending is done using a weighted sum of the original image and the edge image. By adjusting the blending weights, the edges can be emphasized without overpowering the original image. This allows important structural details to stand out while preserving the overall appearance of the image.

## Tracing plt.plot(x, y) in Matplotlib

For this assignment, I used GitHub Copilot to trace what actually happens when calling:
`plt.plot(x, y)`

Even though this looks like one simple function call, Matplotlib actually goes through several layers internally before the graph appears. The call sequence looks like this:
`plt.plot  → gca().plot  → Axes.plot  → add_line`

### plt.plot(x, y)

This is the function we call in our script. It comes from matplotlib.pyplot, this function doesn’t actually draw the line itself. Instead, it acts as a wrapper and passes the work to the current axes object.

### gca().plot(x, y)

gca() stands for **“Get Current Axes.”** This function checks if there is already an axes object in the current figure:
*   If one exists, it returns it.
*   If not, it creates one automatically.

Then it calls the .plot() method on that axes object. This ensures that the data always gets plotted inside a valid graph area.

### Axes.plot(x, y)
This is where most of the real plotting work happens.
Inside this function:
*   The x and y data are checked and processed.
*   A Line2D object is created to represent the line.
*   Styling information (like color or markers) is applied.

This step turns raw numerical data into an actual graphical object.

### add\_line(line)
After creating the Line2D object, Matplotlib needs to attach it to the axes so it can be displayed.
add\_line():
*   Adds the line to the axes’ internal list of drawable objects.
*   Updates axis limits if necessary.
*   Registers the line so it will be rendered when plt.show() is called


## Resources
<img src="Images/OpenCV_logo_black_.png" alt="OpenCV, logo with black text" width="100"/>

[Open-CV](https://opencv.org/)
