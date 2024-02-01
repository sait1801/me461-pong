# Libraries
import cv2
import time
import urllib.request
import numpy as np


class CapturePhoto():

    def __init__(self, camera_url):
        self.camera = camera_url  # IP address of the camera (droidcam)
        self.file_time = time.strftime(
            "%H_%M_%S", time.localtime())  # Current time
        self.file_name = "photos/" + \
            str(self.file_time) + ".jpg"  # File directory

    def capture_photo(self):
        try:
            # Create photo name for each photo captured
            self.file_time = time.strftime(
                "%H_%M_%S", time.localtime())  # Current time
            self.file_name = "photos/" + \
                str(self.file_time) + ".jpg"  # File directory

            # Open the IP camera stream
            stream = urllib.request.urlopen(self.camera)

            # Read the first frame/image
            byte_array = bytearray()
            while True:
                byte_array += stream.read(1024)
                a = byte_array.find(b'\xff\xd8')  # Start of JPEG
                b = byte_array.find(b'\xff\xd9')  # End of JPEG
                if a != -1 and b != -1:
                    jpg = byte_array[a:b + 2]
                    byte_array = byte_array[b + 2:]
                    frame = cv2.imdecode(np.frombuffer(
                        jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    break

            # Save the captured photo
            cv2.imwrite(self.file_name, frame)

            # Return the directory of the captured photo
            return self.file_name

        except Exception as e:
            print(f"Error: {e}")  # Return error


class PhotoLog():

    def __init__(self):
        self.photo_log_path = "logs/photo_log.txt"  # Log directory

    def delete_log(self):
        # Delete whole content of the log file
        try:
            with open(self.photo_log_path, 'w'):
                pass
        except Exception as e:
            print(f"Error: {e}")  # Return error

    # Not needed
    def read_log(self):
        # Read last line of the log file
        try:
            with open(self.photo_log_path, 'r') as file:
                lines = file.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    file.close()
                    return last_line
        except Exception as e:
            return f"Error: {e}"  # Return error

    def write_log(self, photo_dir):
        # Write the directory of the captured photo to the last line of the log file
        try:
            with open(self.photo_log_path, 'a') as file:
                file.write(photo_dir + '\n')
                file.close()
        except Exception as e:
            print(f"Error: {e}")  # Return error


if __name__ == "__main__":
    # Configs
    # todo:uncommnet here ## Replace with the camera's IP address
    camera_url = "http://192.168.30.110:4747/video"

    # Initialize classes
    C = CapturePhoto(camera_url)  # todo:uncommnet here
    P = PhotoLog()

    P.delete_log()  # Delete log every execution

    # Capture photo and log the photo directory every 1 second
    while True:
        phto_dir = C.capture_photo()  # Capture photo todo:uncommnet here
        P.write_log(phto_dir)  # Log photo directory
        time.sleep(1)  # Wait 1 second
