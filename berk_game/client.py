import socket
import pickle  # For serializing game state
import cv2
import mediapipe as mp

# Server settings
HOST = '172.22.64.1'  # The server's IP address
PORT = 1234  # Port to listen on

# Initialize the server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen()

print(f"Server is listening on {HOST}:{PORT}")

# Accept client connections
client_socket, addr = server_socket.accept()
print(f"Connection established with {addr}")

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize game state
game_state = {
    'left_paddle': left_paddle,
    'right_paddle': right_paddle,
    'ball': ball,
    'left_score': left_score,
    'right_score': right_score,
    'ball_direction': ball_direction
}

while True:
    try:
        # Receive hand control data from the client (data representing hand movements)
        received_data = client_socket.recv(1024)
        if received_data:
            hand_control_data = pickle.loads(received_data)

            # Update paddle positions based on hand control data
            # Implement logic to translate hand control data into paddle movements
            # Example: right_paddle.y = hand_control_data * SCALE_FACTOR

        # Serialize and send game state to the client
        game_state_bytes = pickle.dumps(game_state)
        client_socket.send(game_state_bytes)

    except ConnectionResetError:
        print("Client disconnected.")
        break

# Clean up
client_socket.close()
server_socket.close()
