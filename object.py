# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # Load image
# img = cv2.imread('black.jpeg', cv2.IMREAD_COLOR)
# # Convert from cv's BGR default color order to RGB
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.GaussianBlur(img, (5, 5), 0)

# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply threshold
# _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)


# # Create a mask of the same size as the image
# mask = np.zeros((thresh.shape[0] + 2, thresh.shape[1] + 2), dtype=np.uint8)

# # Floodfill parameters
# tolerance = 1  # Tolerance for color differences. Increase for more lenient floodfill
# newVal = (255, 255, 255)  # New color for floodfill area
# loDiff = (tolerance,)*3  # Maximal lower brightness/color difference
# upDiff = (tolerance,)*3  # Maximal upper brightness/color difference


# def fill(event):
#     global img
#     y = event.ydata
#     x = event.xdata

#     # Check if x and y are inside the image
#     if x is not None and y is not None:
#         x, y = int(x), int(y)

#         seed = (x, y)
#         _ = cv2.floodFill(img, mask, seed, newVal, loDiff, upDiff)

#         # Update the image
#         implot.set_data(img)
#         plt.draw()


# # Create a figure and a plot
# fig, ax = plt.subplots()
# implot = ax.imshow(img, origin='upper')

# # Connect the fill function to figure
# fig.canvas.mpl_connect('button_press_event', fill)

# # Show the plot
# plt.show()


from sklearn.cluster import DBSCAN
import cv2
import numpy as np

# Load the image
image = cv2.imread('black.jpeg')

# Reshape the image to be a list of RGB pixels
pixels = image.reshape(-1, 3)

# Perform DBSCAN on the pixels
clustering = DBSCAN(eps=5, min_samples=5).fit(pixels)

# The labels_ property contains the cluster assignments for each pixel
labels = clustering.labels_

# Reshape the labels to have the same shape as the original image
segmented_image = labels.reshape(image.shape[:2])

# Display the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
