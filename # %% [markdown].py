# %% [markdown]
# ##Imports

# %%
from matplotlib.lines import Line2D
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

test = 0

# %% [markdown]
# ##Function Definitions

# %%


def perpendicular_vector(v):
    return [v[0], v[1]]


def point_on_object(point, objects, threshold=10):
    """
    Check if the point is on any of the objects by measuring the distance to each object's points.
    If the distance is less than the threshold, we consider the point to be on the object.

    :param point: The point to check.
    :param objects: The list of objects (clusters of points).
    :param threshold: The distance threshold to consider a point is on the object.
    :return: The index of the object that the point is on, or None if it's on no object.
    """
    for i, obj in enumerate(objects):
        distances = np.linalg.norm(obj - point, axis=1)
        if np.any(distances < threshold):
            return i
    return None


def find_closest_aligned_point(start, direction, objects):
    aligned_points = []
    for obj in objects:
        # Exclude the object that the position lies on
        if any(np.array_equal(point, start) for point in obj):
            continue
        for point in obj:
            # Calculate the y value on the line for the current x value
            y_on_line = (direction[1] / direction[0]) * \
                (point[0] - start[0]) + start[1]
            if np.isclose(point[1], y_on_line, atol=2):  # Adjust the tolerance as needed
                # Create a vector from the start point to the current point
                point_vector = point - start
                # Calculate the dot product between the direction vector and point_vector
                dot_product = np.dot(direction, point_vector)
                # Check if the point is in the direction of the vector
                if dot_product > 0:
                    aligned_points.append(point)

    # If there are no aligned points, return None
    if not aligned_points:
        return None

    # Convert the list of points to an array
    aligned_points = np.array(aligned_points)

    # Find the closest point to the start position in the direction of the vector
    distances = np.linalg.norm(aligned_points - start, axis=1)
    closest_point_index = np.argmin(distances)
    closest_point = aligned_points[closest_point_index]

    return closest_point


def find_normal(points, incoming_vector):
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)

    # Compute the points relative to the centroid
    relative_points = points - centroid

    # Compute the covariance matrix of the relative points
    cov = np.cov(relative_points, rowvar=False)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # The normal vector is the eigenvector corresponding to the smallest eigenvalue
    normal = eigenvectors[np.argmin(eigenvalues)]
    # Swap the components to get (x, y) order
    normal = np.array([-normal[0], normal[1]])

    # Ensure the normal vector points towards the incoming vector
    if np.dot(normal, incoming_vector) > 0:
        normal = -normal  # Reverse the direction of the normal vector

    return normal


# %% [markdown]
# ##Main Running Code
# %%

# Load the image and convert to grayscale for obstacle detection
image = cv2.imread('arena.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray, (5, 5))

binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[
    1]  # Threshold value of 127

print(f"test1 none image: {binary_image.shape !=None}")

# Find the black objects using DBSCAN
y_indices, x_indices = np.nonzero(binary_image == 0)
coordinates = np.column_stack([x_indices, y_indices])
db = DBSCAN(eps=2, min_samples=5).fit(coordinates)
labels = db.labels_

print(f"test2 labels : {labels}")

objects = []
for i in np.unique(labels):
    if i != -1:  # Ignore noise
        objects.append(coordinates[labels == i])

# Define the initial position and vector
x = 0
y = int(binary_image.shape[0] / 2)
dx = 50
dy = 250

# y = int(binary_image.shape[0] / 3)
# dx = 10
# dy = 25
start_position = np.array([x, y], dtype=float)
vector = np.array([dx, dy], dtype=float)

# Normalize the vector
vector = normalize(vector.reshape(1, -1))[0]

# Compute the normal vectors for each object
# normals = [find_normal(obj) for obj in objects] //todo
normals = []


# Find the index of the object that contains the start position
start_object_index = next(i for i, obj in enumerate(
    objects) if start_position in obj)

# Define the center and radius of the circular area
center = np.array([binary_image.shape[1] // 2, binary_image.shape[0] // 2])
radius = 10

# Initialize the total length
total_length = 0

positions = []
for _ in range(5):
    print(f"ITERATION : {_}")
    closest_aligned_point = find_closest_aligned_point(
        start_position, vector, objects)
    print(f"init vector:{vector}")

    if closest_aligned_point is None:
        # If there are no aligned points, break the loop
        print("No Aligned Points ! ")
        break

    # Calculate the distance to the closest aligned point
    distance = np.linalg.norm(closest_aligned_point - start_position)
    total_length += distance

    # WINNER CONDITION
    if np.linalg.norm(start_position - center) <= radius:
        # If it is, break the loop and return the initial starting position, the angle of the vector, and the total length
        angle = np.arctan2(vector[1], vector[0])
        print("##################################### WINNER CONDITION #####################################")
        print(
            f"Initial starting position: {start_position}, Angle of the vector: {angle}, Length: {total_length}")
        break

    # Move to the closest aligned point
    start_position = closest_aligned_point

    start_object_index = point_on_object(start_position, objects)

    # Assuming normals[start_object_index] is already a normalized normal vector
    normal = find_normal(objects[start_object_index], vector)

    normals.append(normal)

    # Calculate the dot product of the vector and the normal
    dot_prod = np.dot(vector, normal)

    # Calculate the reflected vector
    reflected_vector = vector - 2 * dot_prod * normal

    if test:
        positions.append(start_position)

        print(f"start_object_index: {start_object_index}")

        print(f"NORMAL: {normal}")

        print(f"refelcted_vector : {reflected_vector}")

        # Create custom legend entries
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Starting Position',
                   markerfacecolor='red', markersize=10),
            Line2D([0], [0], color='yellow', lw=4, label='Initial Vector'),
            Line2D([0], [0], color='blue', lw=4, label='Reflected Vector'),
            Line2D([0], [0], color='green', lw=4, label='Normal Vector')
        ]

        # Add the legend to the plot with custom entries
        plt.legend(handles=legend_elements, loc='upper right')

        # Draw the point and vector on the image
        plt.imshow(binary_image, cmap='gray')
        plt.scatter(start_position[0], start_position[1], color='red')
        plt.arrow(start_position[0], start_position[1], vector[0]
                  * 50, vector[1]*50, color='yellow', head_width=3)
        plt.arrow(start_position[0], start_position[1], reflected_vector[0]
                  * 50, reflected_vector[1]*50, color='blue', head_width=3)
        # Draw the normal vector of the object that the start position is on
        plt.arrow(start_position[0], start_position[1], normals[_]
                  [0]*50, normals[_][1]*50, color='green', head_width=3)
        plt.pause(0.001)  # Pause to update the figure

    # Update the vector to the reflected vector for the next iteration
    vector = reflected_vector

if test:
    # Keep the plot open
    plt.show()


# %%
image_rgb = cv2.imread('arena.png')

# Convert the image to RGB (matplotlib expects RGB format)

# Create a figure and axis to plot on
fig, ax = plt.subplots()

# Display the image
ax.imshow(image_rgb)

# Loop through the positions and plot each one
for i, position in enumerate(positions):
    # Scatter plot for the point
    ax.scatter(position[0], position[1], color='red',
               s=50)  # s is the size of the point

    # Annotate the point with its number
    ax.annotate(str(i + 1), (position[0], position[1]),
                textcoords="offset points", xytext=(0, 10), ha='center', color='blue')

    # If not the first point, draw a line from the previous point
    if i > 0:
        # Get the previous point
        prev_position = positions[i - 1]
        # Draw a line from the previous point to the current point
        ax.plot([prev_position[0], position[0]], [
                prev_position[1], position[1]], color='green')

# Show the plot
plt.show()

# %% [markdown]
# ##Itreration 360x800x5

# %%
print(f"Number of Iterations: {round(image.shape[0]/10)* 5* 360} ")

# %%
# Assuming all the necessary functions and imports are already defined above this code

# Define the center and radius of the circular area
center = np.array([binary_image.shape[1] // 2, binary_image.shape[0] // 2])
radius = 10

# List to store the results
winning_conditions = []

# Iterate over y values from 0 to image height, incrementing by 10 pixels
for y in range(0, binary_image.shape[0], 10):
    print(f"New Y iter {y}")
    # Iterate over angles from 0 to 360 degrees
    for angle in range(0, 360):
        print(f"New angle iter {angle}")

        # Convert angle to radians and calculate dx, dy components of the unit vector
        radians = np.radians(angle)
        dx = np.cos(radians)
        dy = np.sin(radians)

        # Define the initial position and vector
        start_position = np.array([0, y], dtype=float)
        vector = np.array([dx, dy], dtype=float)

        # Normalize the vector
        vector = vector / np.linalg.norm(vector)

        # Initialize the total length and number of bounces
        total_length = 0
        bounces = 0

        # Perform up to 5 iterations
        for _ in range(5):
            closest_aligned_point = find_closest_aligned_point(
                start_position, vector, objects)

            if closest_aligned_point is None:
                # If there are no aligned points, break the loop
                break

            # Calculate the distance to the closest aligned point
            distance = np.linalg.norm(closest_aligned_point - start_position)
            total_length += distance
            bounces += 1

            # WINNER CONDITION
            if np.linalg.norm(start_position - center) <= radius:
                # Record the winning condition
                winning_conditions.append({
                    'start_position': start_position.tolist(),
                    'angle': angle,
                    'length': total_length,
                    'bounces': bounces
                })
                break

            # Move to the closest aligned point
            start_position = closest_aligned_point

            # Reflect the vector
            normal = find_normal(
                objects[point_on_object(start_position, objects)], vector)
            dot_prod = np.dot(vector, normal)
            reflected_vector = vector - 2 * dot_prod * normal

            # Update the vector to the reflected vector for the next iteration
            vector = reflected_vector

# Print or process the winning conditions as needed
for condition in winning_conditions:
    print(condition)


# %%
