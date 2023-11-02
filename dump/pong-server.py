import socket
import threading
import json

# Game state
game_state = {
    "paddle1": 50,  # Paddle position
    "paddle2": 50,
    "ball": (100, 100),
    "score": [0, 0]
}

# Lock for managing game state access
state_lock = threading.Lock()


def handle_client(conn, player):
    global game_state
    while True:
        try:
            # Receive data from client
            data = conn.recv(1024).decode()
            if not data:
                break

            # Update game state based on received data
            with state_lock:
                if player == 1:
                    game_state["paddle1"] = int(data)
                else:
                    game_state["paddle2"] = int(data)

                # Send updated game state to client
                conn.sendall(json.dumps(game_state).encode())

        except Exception as e:
            print(f"Error: {e}")
            break

    conn.close()


def start_server():
    host = '172.22.64.1'
    port = 5555

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen()

    print("Server started. Waiting for connections...")
    player = 0

    while True:
        conn, addr = server.accept()
        print(f"Connected by {addr}")
        player += 1
        thread = threading.Thread(target=handle_client, args=(conn, player))
        thread.start()


if _name_ == "_main_":
    start_server()
