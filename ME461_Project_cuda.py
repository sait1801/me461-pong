# Libraries
import time
import serial
# import serial.tools.list_ports //todo uncomment here
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QPixmap

import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cProfile
from numba import cuda

import numpy as np
import numba.cuda as cuda


@cuda.jit
def find_closest_aligned_point_kernel(start, direction, objects, results):
    i = cuda.grid(1)
    if i < objects.shape[0]:
        point = objects[i]
        y_on_line = (direction[1] / direction[0]) * \
            (point[0] - start[0]) + start[1]
        if abs(point[1] - y_on_line) <= 2:
            point_vector = cuda.local.array(point.shape, point.dtype)
            for i in range(point.shape[0]):
                point_vector[i] = point[i] - start

            dot_product = np.dot(point_vector, direction)
            if dot_product > 0:
                distance = np.linalg.norm(point_vector)
                results[i] = distance, point  # Store distance and point


def find_closest_aligned_point(start, direction, objects):
    objects = np.array(objects)
    start = np.array(start)
    direction = np.array(direction)
    all_points = np.concatenate(objects)

    objects_device = cuda.to_device(all_points)
    results_device = cuda.device_array_like(all_points)

    threads_per_block = 32
    blocks_per_grid = (
        objects.shape[0] + threads_per_block - 1) // threads_per_block
    find_closest_aligned_point_kernel[blocks_per_grid, threads_per_block](
        start, direction, objects_device, results_device)

    results = results_device.copy_to_host()
    closest_distance, closest_point = np.amin(results, axis=0)
    return closest_point


class Detection():

    def __init__(self):
        self.is_reference_detected = False
        self.is_robot_detected = False
        self.is_circle_detected = False
        self.is_color_detected = False
        self.is_starter_puck_placed = False
        self.is_opponent_done = False
        self.is_our_pos_detected = False
        self.is_on_the_left = False
        self.is_on_the_right = False
        self.puck_color = None

    def DetectReference(self, photo_dir):
        """
        This function finds how many pixels correspond to how many cm in the x and y plane.
        Use the output of this function in pathfinding and related algorithms.

        :param photo_dir (directory of the captured photo)
        :return ratio_x, ratio_y (ratios), is_reference_detected
        """

        try:
            # Known lengths
            # length of one of the aluminum profile
            self.ref_x_cm = 90  # cm
            # distance between 2 aluminum profile
            self.ref_y_cm = 60  # cm

            image = cv2.imread(photo_dir)

            """
            You should find these distances in terms of pixel using image detection
            TODO: Manually find this by measuring it. 
            """
            # Random values are given to test (find them)
            self.ref_x_px = image.shape[1]  # pixel
            self.ref_y_px = image.shape[0]  # pixel

            # Pixel per cm ratios for x and y axis
            self.ratio_x = self.ref_x_px / self.ref_x_cm  # pixel per cm
            self.ratio_y = self.ref_y_px / self.ref_y_cm  # pixel per cm

            # To check whether references are detected
            self.is_reference_detected = True

            return self.ratio_x, self.ratio_y, self.is_reference_detected  # Return ratios

        # References are not detected
        except Exception:
            print("Exception in DetectReference func")

            self.is_reference_detected = False
            return -1, -1, self.is_reference_detected

    def DetectMe(self, photo_dir):
        """
        Detects on which side our robot is by finding the QR code.

        :param photo_dir: directory of the captured photo
        :return: is_on_the_left, is_on_the_right, is_robot_detected
        """
        try:
            # Load the image from the given photo directory
            image = cv2.imread(photo_dir)

            # Initialize a QR Code detector
            qr_detector = cv2.QRCodeDetector()

            # Detect the QR code in the image
            data, bbox, _ = qr_detector.detectAndDecode(image)

            # Check if a QR code has been detected
            if bbox is not None:
                # Calculate the center of the QR code
                center_x = np.mean(bbox[0][:, 0])

                # Get the width of the image
                image_width = image.shape[1]

                # Determine on which side of the image the QR code is
                if center_x < image_width / 2:
                    self.is_on_the_left = True
                    self.is_on_the_right = False
                else:
                    self.is_on_the_left = False
                    self.is_on_the_right = True

                # Set the robot detected flag to True
                self.is_robot_detected = True
            else:
                # If QR code is not detected, set all flags to False
                self.is_on_the_left = False
                self.is_on_the_right = False
                self.is_robot_detected = False

            # Return the side detections and the robot detected flag
            return self.is_on_the_left, self.is_on_the_right, self.is_robot_detected

        except Exception as e:
            # In case of an error, set all flags to False and return
            print("Exception in DetectMe func")

            self.is_on_the_left = False
            self.is_on_the_right = False
            self.is_robot_detected = False
            return self.is_on_the_left, self.is_on_the_right, self.is_robot_detected

    def DetectCircle(self, photo_dir):
        """
        This function detects the position of a reddish circle and its centroid.

        :param photo_dir: directory of the captured photo
        :return: circle_pos_x, circle_pos_y, is_circle_detected
        """
        try:
            # Load the image
            image = cv2.imread(photo_dir)

            # Convert the image to the HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define the color range for red
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])

            # Threshold the image to get only red colors
            mask = cv2.inRange(hsv, lower_red, upper_red)

            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Assume the largest contour is the reddish circle
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                # Set the circle detected flag to True
                self.is_circle_detected = True
                self.circle_pos_x = cX
                self.circle_pos_y = cY

                # Optionally, draw the contour and centroid on the image
                # cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 2)
                # cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)

                # Return the centroid coordinates and the circle detected flag
                return self.circle_pos_x, self.circle_pos_y, self.is_circle_detected

            # If no contours are found, set the flag to False
            else:
                self.is_circle_detected = False
                return -1, -1, self.is_circle_detected

        except Exception as e:
            # In case of an error, set the flag to False and return
            print("Exception in DetectCircle func")
            self.is_circle_detected = False
            return -1, -1, self.is_circle_detected

    def DetectPuckColor(self, photo_dir, x1, y1, x2, y2):
        """
        This function detects the average color of the puck within a specified region.

        :param photo_dir: directory of the captured photo
        :param x1, y1: coordinates of the first point
        :param x2, y2: coordinates of the second point
        :return: puck_color, is_color_detected
        """

        try:
            # Load the image
            image = cv2.imread(photo_dir)

            # Crop the region of interest based on the provided coordinates
            roi = image[y1:y2, x1:x2]

            # Convert the cropped image to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Define a mask for the circular puck using color thresholding
            # Note: The color range may need to be adjusted based on the puck's color
            lower_color = np.array([0, 50, 50])
            upper_color = np.array([10, 255, 255])
            mask = cv2.inRange(hsv_roi, lower_color, upper_color)

            # Calculate the average color of the puck
            # Note: cv2.mean returns a tuple of 4 values (mean of B, G, R, and A channels)
            # We only need the BGR values, hence we take the first 3 values
            mean_val = cv2.mean(roi, mask=mask)[:3]

            # Convert the BGR color to RGB color
            puck_color = (mean_val[2], mean_val[1], mean_val[0])

            # Set the color detected flag to True
            self.is_color_detected = True
            self.puck_color = puck_color

            # Return the average puck color and the color detected flag
            # todo: opponent puck color finde rekle amk
            return puck_color, (0, 0, 0), self.is_color_detected

        except Exception as e:
            # In case of an error, set the flag to False and return a default color
            print("Error in DetectPuckColor function")
            self.is_color_detected = False
            return (0, 0, 0), (0, 0, 0), self.is_color_detected

    def DetectStarterPuck(self, photo_dir):
        """
        This function checks for the presence of a starter puck at the mid-bottom of the image and determines its color.

        :param photo_dir: directory of the captured photo
        :return: starter_puck_color, is_starter_puck_placed

        todo: make the gui a selector rectangle for the starter puck 
        """
        try:
            # Load the image
            image = cv2.imread(photo_dir)

            # Calculate the puck's expected position and size based on the image dimensions
            image_height, image_width, _ = image.shape
            # Assuming ratio_x is already set by DetectReference method
            puck_diameter_px = self.ratio_x
            # print(f"puck_diameter_px : {puck_diameter_px}")
            puck_radius_px = puck_diameter_px / 2
            mid_bottom_x = image_width // 2
            mid_bottom_y = image_height - puck_diameter_px

            # Define the region of interest for the puck's expected location
            # roi = image[int(mid_bottom_y - puck_radius_px):int(mid_bottom_y + puck_radius_px),
            #             int(mid_bottom_x - puck_radius_px):int(mid_bottom_x + puck_radius_px)]

            roi = image[int(1150):int(1230),
                        int(1485):int(1603)]

            # Convert the region of interest to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Define a mask to isolate the puck based on color
            # Note: The color range should be adjusted based on the lighting and puck color
            lower_color = np.array([0, 0, 200])
            # Left here , detect the starting puck color amk
            upper_color = np.array([180, 255, 255])
            mask = cv2.inRange(hsv_roi, lower_color, upper_color)

            # Check if there is a puck in the region of interest
            if cv2.countNonZero(mask) > 0:
                # Calculate the average color of the puck
                mean_color = cv2.mean(hsv_roi, mask=mask)
                self.starter_puck_color = mean_color[:3]  # Get the BGR values
                self.is_starter_puck_placed = True
            else:
                # Default color if no puck is detected
                self.starter_puck_color = (0, 0, 0)
                self.is_starter_puck_placed = False

            return self.starter_puck_color, self.is_starter_puck_placed

        except Exception as e:
            # In case of an error, set the color to default and the flag to False
            print(f"Error in DetectStarterPuck function : {e}")
            self.starter_puck_color = (0, 0, 0)
            self.is_starter_puck_placed = False
            return self.starter_puck_color, self.is_starter_puck_placed

    def DetectPosOurPuck(self, photo_dir):
        """
        This function detects our puck in the left region of interest of the image.

        :param photo_dir: directory of the captured photo
        :return: our_puck_pos_x, our_puck_pos_y, is_our_puck_detected
        """
        try:
            # Load the image
            image = cv2.imread(photo_dir)

            # Define the region of interest (ROI) width based on the puck's diameter (2 cm to pixels)
            # Assuming ratio_x is already set by DetectReference method
            roi_width = int(2 * self.ratio_x)

            # Define the left region of interest in the image
            roi = image[:, :roi_width]

            # Convert the ROI to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Convert our puck color from RGB to HSV
            hsv_puck_color = cv2.cvtColor(
                np.uint8([[self.puck_color]]), cv2.COLOR_RGB2HSV)[0][0]

            # Define a small range around the puck color for HSV thresholding
            sensitivity = 15  # Sensitivity range can be adjusted
            lower_color = np.array(
                [hsv_puck_color[0] - sensitivity, hsv_puck_color[1] - sensitivity, hsv_puck_color[2] - sensitivity])
            upper_color = np.array(
                [hsv_puck_color[0] + sensitivity, hsv_puck_color[1] - sensitivity, hsv_puck_color[2] - sensitivity])

            # Threshold the image to get only the colors of our puck
            mask = cv2.inRange(hsv_roi, lower_color, upper_color)

            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Assume the largest contour is our puck
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                # Set the puck position detected flag to True
                self.is_our_puck_detected = True
                self.our_puck_pos_x = cX
                self.our_puck_pos_y = cY
            else:
                # If no puck is detected, set the flag to False
                self.is_our_puck_detected = False
                self.our_puck_pos_x = -1
                self.our_puck_pos_y = -1
            return 672, 573, True  # todo? buraya bakarlar amk sil burayi
            return self.our_puck_pos_x, self.our_puck_pos_y, self.is_our_puck_detected

        except Exception as e:
            # In case of an error, set the flag to False and return default values
            print("Error in DetectPosOurPuck function")
            self.is_our_puck_detected = False
            return -1, -1, self.is_our_puck_detected

    def DetectOpponentTurn(self, photo_dir, opponent_puck_number):
        self.opponent_puck_number = opponent_puck_number
        self.opponent_puck_number += 1
        self.opponent_current_puck_position_x = 1020  # pixel
        self.opponent_current_puck_position_y = 420  # pixel
        self.is_opponent_turn_done = True
        return self.opponent_puck_number, self.is_opponent_turn_done

    def DetectOurTurn(self, photo_dir, our_puck_number):
        self.our_puck_number = our_puck_number
        self.our_puck_number += 1
        self.is_our_turn_done = True
        return self.our_puck_number, self.is_our_turn_done


class PathFinder():

    def __init__(self, ratio_x, ratio_y, circle_pos_x, circle_pos_y, robot_side):
        self.ratio_x = ratio_x
        self.ratio_y = ratio_y
        self.circle_pos_x = circle_pos_x
        self.circle_pos_y = circle_pos_y
        self.robot_side = robot_side

    def binarize_image(self, image_path):
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Convert the image to grayscale

        # Binarize the image: keep black pixels black and turn other pixels white
        _, binary_image = cv2.threshold(image, 65, 255, cv2.THRESH_BINARY)

        return binary_image

    def find_objects_with_dbscan(self, binary_image, eps=2, min_samples=5):
        # Find the black pixels in the binary image
        # Since we inverted the image, black is now 255
        y_indices, x_indices = np.nonzero(binary_image == 255)
        coordinates = np.column_stack([x_indices, y_indices])

        # Apply DBSCAN to the coordinates of the black pixels
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
        labels = db.labels_

        # Group the coordinates by their cluster label
        unique_labels = set(labels)
        clusters = [coordinates[labels == label]
                    for label in unique_labels if label != -1]

        return clusters

    def FindPath(self, binary_image, objects, centre_positions):
        # Assuming DetectCircle has been called and self.circle_pos_x, self.circle_pos_y have been set
        center = np.array(
            [centre_positions[1], centre_positions[0]])  # in X,Y order
        radius = 3  # Radius can be adjusted based on the specific game rules

        # List to store the results
        winning_conditions = []

        # Iterate over y values from 0 to image height, incrementing by 10 pixels
        for y in range(0, binary_image.shape[0], int(binary_image.shape[0] / 20)):
            print(f"Y iter: {y}")
            # Iterate over angles from 0 to 360 degrees
            for angle in range(-90, 90, 10):
                radians = np.radians(angle)
                dx = np.cos(radians)
                dy = np.sin(radians)

                start_position = np.array([0, y], dtype=float)
                vector = np.array([dx, dy], dtype=float)
                vector = vector / np.linalg.norm(vector)

                total_length = 0
                bounces = 0
                point_vectors = []

                for _ in range(5):
                    point_vectors.append(start_position)
                    closest_aligned_point = find_closest_aligned_point(
                        start=start_position, direction=vector, objects=objects)

                    if closest_aligned_point is None:
                        break

                    distance = np.linalg.norm(
                        closest_aligned_point - start_position)
                    total_length += distance
                    bounces += 1

                    direction_unit_vector = vector / np.linalg.norm(vector)
                    start_to_center = center - start_position
                    projection_length = np.dot(
                        start_to_center, direction_unit_vector)
                    projection_vector = projection_length * direction_unit_vector
                    center_to_line = start_to_center - projection_vector
                    distance_to_line = np.linalg.norm(center_to_line)

                    if distance_to_line <= radius:
                        winning_conditions.append({
                            'start_position': start_position.tolist(),
                            'angle': angle,
                            'length': total_length,
                            'bounces': bounces,
                            'points': point_vectors,
                            'fin_vector': vector
                        })
                        break

                    start_position = closest_aligned_point
                    normal = self.find_normal(
                        objects[self.point_on_object(start_position, objects)], vector)
                    dot_prod = np.dot(vector, normal)
                    reflected_vector = vector - 2 * dot_prod * normal
                    vector = reflected_vector

        return winning_conditions

    def Path2Variable(self, path):
        self.pwm_duty = 500
        self.pwm_freq = 1000
        self.robot_angle = 45
        length = path['length']
        bounde_rate = path['bounces']  # todo set pwm here =???dogukan
        return str(self.pwm_duty), str(self.pwm_freq), self.robot_angle

    def DrawPath(self, photo_dir):
        pass

    def perpendicular_vector(self, v):
        return [v[1], -v[0]]

    def point_on_object(self, point, objects, threshold=10):
        for i, obj in enumerate(objects):
            distances = np.linalg.norm(obj - point, axis=1)
            if np.any(distances < threshold):
                return i
        return None

    def plot_path_on_image(self, image_path, path_data, circle_pos_x, circle_pos_y):
        # Load the binarized image
        binary_image = self.binarize_image(image_path)
        # time.sleep(10)

        # Plot the binarized image
        plt.imshow(binary_image, cmap='gray')

        # Extract the points and final vector from the path data
        points = path_data['points']
        final_vector = path_data['fin_vector']
        length = path_data['length']

        # Plot the points and the lines between them
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'ro-')
            plt.text(start[0], start[1], str(i + 1), fontsize=12, ha='right')

        # Label the last point
        plt.text(points[-1][0], points[-1][1],
                 str(len(points)), fontsize=12, ha='right')

        # Draw the final vector from the last point
        final_point = points[-1] + length * final_vector * 0.25
        plt.arrow(points[-1][0], points[-1][1], final_point[0] - points[-1][0],
                  final_point[1] - points[-1][1], head_width=10, head_length=10, fc='green', ec='green')

        # Draw a blue circle at the centroid
        circle = plt.Circle((circle_pos_x, circle_pos_y),
                            10, color='blue', fill=True)
        plt.gca().add_patch(circle)

        # Display the plot
        plt.axis('off')  # Turn off the axis
        plt.show()

        time.sleep(10)  # todo: amk

    def find_normal(self, points, incoming_vector):
        centroid = np.mean(points, axis=0)
        relative_points = points - centroid
        cov = np.cov(relative_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        normal = np.array([normal[0], normal[1]])
        if np.dot(normal, incoming_vector) > 0:
            normal = -normal
        return normal


class PhotoLog():

    def __init__(self):
        self.photo_log_path = "logs/photo_log.txt"

    # Not needed

    def delete_log(self):
        try:
            with open(self.photo_log_path, 'w'):
                pass
        except Exception as e:
            print(f"Error: {e}")

    def read_log(self):
        try:
            with open(self.photo_log_path, 'r') as file:
                lines = file.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    file.close()
                    return last_line
        except Exception as e:
            return f"Error: {e}"

    # Not needed

    def write_log(self, photo_dir):
        try:
            with open(self.photo_log_path, 'a') as file:
                file.write(photo_dir + '\n')
                file.close()
        except Exception as e:
            print(f"Error: {e}")


class Gui(QMainWindow):

    def __init__(self):
        super(Gui, self).__init__()

        self.setWindowTitle("Score Board")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

    def angle_gui(self, photo_dir, robot_angle):
        # Load the image
        self.pixmap = QPixmap(photo_dir)

        # Display the image using QLabel
        self.image_label = QLabel(self)
        self.image_label.setPixmap(self.pixmap)
        self.layout.addWidget(self.image_label)

        # Create a QPushButton
        button = QPushButton('Ready', self)

        # Connect the button's clicked signal to the close_window method
        button.clicked.connect(self.ready)

        self.show()

    def ready(self):
        # Close the window when the button is pressed
        self.close()


class Transmitter:
    TERMINATOR = '\r'.encode('UTF8')

    def __init__(self, timeout=1):
        self.ports = serial.tools.list_ports.comports(include_links=False)
        if len(self.ports) == 1:
            self.port = self.ports[0].device
            self.serial = serial.Serial(
                port=self.port, baudrate=115200, timeout=timeout)
            print("Succesfully connected to Pico via " + self.port + "\n")
        elif len(self.ports) == 0:
            print("Pico could not found")
        else:
            self.portlist = ""
            for i in range(len(self.ports)):
                self.portlist += self.ports[i].device + "\n"
            self.port = input(
                "There are many COM devices. Please write COM port of the Pico.\n" + self.portlist)
            self.serial = serial.Serial(
                port=self.port, baudrate=115200, timeout=timeout)
            print("Succesfully connected to Pico via " + self.port + "\n")

    def wait_user(self):
        self.send("start!")
        self.receive()
        while True:
            if self.receive() == "OK!":
                break

    def send(self, text: str):
        line = '%s\r\f' % text
        self.serial.write(line.encode('utf-8'))

    def receive(self) -> str:
        line = self.serial.read_until(self.TERMINATOR)
        return line.decode('UTF8').strip()

    def close(self):
        self.serial.close()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    # Initialize classes
    D = Detection()
    PL = PhotoLog()
    # T = Transmitter()
    app = QApplication(sys.argv)

    time.sleep(3)  # Wait (until first photo captured)

    while True:
        # Take directory of the captured photo
        # photo_dir = PL.read_log() //todo: uncommnet here
        photo_dir = 'arena3_edited.jpeg'

        # Find pixel per cm ratio for x and y axis
        ratio_x, ratio_y, is_reference_detected = D.DetectReference(photo_dir)

        # Find our robot is on which side
        is_on_the_left, is_on_the_right, is_robot_detected = D.DetectMe(
            photo_dir)

        print(
            f"TEST1 - Side : rigth:{is_on_the_right}, left : {is_on_the_left}")

        # Find the position of the circle
        circle_pos_x, circle_pos_y, is_circle_detected = D.DetectCircle(
            photo_dir)

        print(f"TEST2 -Circle Centroid : {circle_pos_x,circle_pos_y}")

        print("here1")

        # If everything needed is detected break the loop. Otherwise, try to detect again.
        if is_reference_detected and is_robot_detected and is_circle_detected:
            break
    print("here2")

    print(f"TEST3 -Reference ratio x,y: {ratio_x,ratio_y}")

    # Robot side
    if is_on_the_left:
        robot_side = "left"
    elif is_on_the_right:
        robot_side = "right"

    print(f"TEST4 -Robot Side: {robot_side}")

    while True:
        # Take directory of the captured photo
        # photo_dir = PL.read_log() //photo dir will be the image
        photo_dir = 'arena3_edited.jpeg'
        print("here3")

        # Find the color of the pucks
        our_puck_color, opponent_puck_color, is_color_detected = D.DetectPuckColor(
            photo_dir, 1490, 1146, 1620, 1272)

        # If everything needed is detected break the loop. Otherwise, try to detect again.
        if is_color_detected:
            break
    print(
        f"TEST5 - Puck Color: ours: {our_puck_color}, opponents: {opponent_puck_color}")

    # Initialize PathFinder using ratio_x, ratio_y, circle_pos_x, circle_pos_y and robot_side
    PF = PathFinder(ratio_x, ratio_y, circle_pos_x, circle_pos_y, robot_side)

    print("here4")

    while True:
        # Take directory of the captured photo
        # photo_dir = PL.read_log() // todo
        photo_dir = 'arena3_edited.jpeg'

        # If color needed
        color_needed = True

        # Check whether the game starter puck is placed and find its color
        starter_puck_color, is_starter_puck_placed = D.DetectStarterPuck(
            photo_dir)

        # Wait until the game starter puck is placed.
        if is_starter_puck_placed:
            break
    print("here5")
    print(f"TEST6 - Starter Color:  {starter_puck_color}")

    # Color not needed after one time finded
    color_needed = False

    # Comparison of puck colors
    our_color_distance = np.linalg.norm(
        np.array(our_puck_color) - np.array(starter_puck_color))
    opponent_color_distance = np.linalg.norm(
        np.array(opponent_puck_color) - np.array(starter_puck_color))

    if our_color_distance < opponent_color_distance:  # Our turn
        is_our_turn = True
        is_opponent_turn = False
    elif our_color_distance > opponent_color_distance:  # Opponent turn
        is_our_turn = False
        is_opponent_turn = True

        # Number of the pucks on the arena
    our_puck_number = 0
    opponent_puck_number = 0

    # Positions of the pucks on the arena
    our_puck_positions = []
    opponent_puck_positions = []
    print("here6")

    counter = 0  # todo remove this
    while True:

        if is_our_turn == True and is_opponent_turn == False:
            while True:
                # Take directory of the captured photo
                # photo_dir = PL.read_log()
                photo_dir = 'arena3_edited.jpeg'

                # Find the position of our puck
                our_puck_pos_x, our_puck_pos_y, is_our_pos_detected = D.DetectPosOurPuck(
                    photo_dir)

                # If everything needed is detected break the loop. Otherwise, try to detect again.
                if is_our_pos_detected:
                    break

            print("here7")

            # Find the first path
            # binarized image, objects, centroid of circle
            # Binarize the image
            # todo: path find için sağdan ve soldan mekanizmaları kesmen gerekiyor
            circle_pos_x, circle_pos_y, is_circle_detected = D.DetectCircle(
                'path_find_arena3.jpg')

            print(f"new circle coords : {circle_pos_x,circle_pos_y}")

            binarized_image = PF.binarize_image('path_find_arena3.jpg')
            print("here7.555")

            # cv2.imshow('Binary Image', binarized_image)

            # # Display the image in a window
            # cv2.imshow('Window Title', binarized_image)

            # # Wait for any key to be pressed before closing the window
            # cv2.waitKey(0)

            # # Destroy all windows
            # cv2.destroyAllWindows()

            # time.sleep(30)

            # Find objects using DBSCAN
            clusters = PF.find_objects_with_dbscan(
                binary_image=binarized_image)
            print(f"here7.8, shape of image : {binarized_image.shape}")
            # Find the position of the circle

            path = PF.FindPath(binarized_image, clusters,
                               (circle_pos_x, circle_pos_y))

            print("here7.9s")

            if counter == 0:
                counter += 1
                print(f"TEST7 - Path: {path[0]}")

            # Test code
            PF.plot_path_on_image(
                'path_find_arena3.jpg', path[3], circle_pos_x=circle_pos_x, circle_pos_y=circle_pos_y)

            # Convert path knowledge to the variables to be sent to Pico
            pwm_duty, pwm_freq, robot_angle = PF.Path2Variable(path[3])

            # Draw path to captured photo
            PF.DrawPath(photo_dir)
            print("here8")

            # Show path and angle on the Gui
            G = Gui()
            print("here9")

            G.angle_gui(photo_dir, robot_angle)
            print("here10")

            while True:
                # Take directory of the captured photo
                # photo_dir = PL.read_log() // todo
                photo_dir = 'arena3_edited.jpeg'

                # Check whether the game starter puck is placed
                garbage1, is_starter_puck_placed = D.DetectStarterPuck(
                    photo_dir)

                # Wait until the game starter puck is placed again.
                if is_starter_puck_placed:
                    break

            # T.send(pwm_duty + ":" + pwm_freq) #todo: uncomment here

            while True:
                # Take directory of the captured photo
                # photo_dir = PL.read_log() // todo: uncommnet here
                photo_dir = 'arena3_edited.jpeg'

                # Is our turn done
                our_puck_number, is_our_turn_done = D.DetectOurTurn(
                    photo_dir, our_puck_number)

                if is_our_turn_done:
                    is_our_turn = False
                    is_opponent_turn = True
                    break

        elif is_our_turn == False and is_opponent_turn == True:
            while True:
                # Take directory of the captured photo
                # photo_dir = PL.read_log() // todo: uncomment here
                photo_dir = 'arena3_edited.jpeg'

                # Is opponent turn done
                opponent_puck_number, is_opponent_turn_done = D.DetectOpponentTurn(
                    photo_dir, opponent_puck_number)

                if is_our_turn_done:
                    is_our_turn = True
                    is_opponent_turn = False
                    break

        if our_puck_number == 5 and opponent_puck_number == 5:
            break
    profiler.disable()
    profiler.print_stats(sort='time')
    sys.exit(app.exec_())
