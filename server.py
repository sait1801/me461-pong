import socket
from _thread import *
import sys

server = "localhost"
port = 5555

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    s.bind((server, port))
except socket.error as e:
    str(e)

s.listen(2)
print("Waiting for a connection, Server Started")


def read_pos(str):
    str = str.split(",")
    return int(str[0]), int(str[1]), int(str[2]), int(str[3])


def make_pos(tup):
    return str(tup[0]) + "," + str(tup[1]) + "," + str(tup[2]) + "," + str(tup[3])


pos = [(50, 150), (900, 150)]
ball_pos = (250, 500)


def threaded_client(conn, player, ball_pos_thread):
    # print("player")
    # print(player)
    pos_tuple = (pos[player][0], pos[player][1],
                 ball_pos_thread[0], ball_pos_thread[1])
    conn.send(str.encode(make_pos(pos_tuple)))

    reply = ""
    while True:
        try:
            data = read_pos(conn.recv(2048).decode())
            pos[player] = data[:2]
            ball_pos = data[2:]

            # print(f"player_pos = {pos[player]}")
            # print(f"ball_pos = {ball_pos}")

            if not data:
                # print("Disconnected")
                break
            else:
                if player == 1:
                    reply = (pos[0][0], pos[0]
                             [1], ball_pos[0], ball_pos[1])
                    # reply = pos[0]
                else:
                    reply = (pos[1][0], pos[1]
                             [1], ball_pos[0], ball_pos[1])
                    # reply = pos[1]

                # print("Received: ", data)
                # print("Sending : ", reply)

            conn.sendall(str.encode(make_pos(reply)))
        except:
            break

    print("Lost connection")
    conn.close()


currentPlayer = 0
while True:
    try:
        conn, addr = s.accept()
        print("Connected to:", addr)

        if currentPlayer > 2:
            print("Players at max number ")
        else:
            start_new_thread(threaded_client, (conn, currentPlayer, ball_pos))
            currentPlayer += 1
    except:
        print("Players at max number ")
        pass
