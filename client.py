import pygame
from ball import Ball
from network import Network
from player import Player

width = 1200
height = 600
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Client")

clientNumber = 0


def read_pos(str):
    str = str.split(",")
    return int(str[0]), int(str[1]), int(str[2]), int(str[3])


def make_pos(tup):
    # print("herwe")
    # print(tup)
    return str(tup[0]) + "," + str(tup[1]) + "," + str(tup[2]) + "," + str(tup[3])


def redrawWindow(win, player, player2, ball):
    win.fill((0, 0, 0))
    player.draw(win)
    player2.draw(win)
    ball.draw(win)
    pygame.display.update()


def main():
    run = True
    n = Network()
    startPos = read_pos(n.getPos())
    p = Player(startPos[0], startPos[1], 10, 200, (0, 255, 0))
    p2 = Player(0, 900, 10, 200, (255, 0, 0))
    ball = Ball(posx=width/2, posy=height/2,
                speed=(10, 0), color=(255, 80, 128))
    clock = pygame.time.Clock()

    while run:
        clock.tick(60)
        p2Pos = read_pos(
            n.send(make_pos((p.x, p.y, int(ball.posx), int(ball.posy)))))
        p2.x = p2Pos[0]
        p2.y = p2Pos[1]

        ball.posx = p2Pos[2]
        ball.posy = p2Pos[3]

        p2.update()
        ball.move()
        collide1 = pygame.Rect.colliderect(ball.rect, p.rect)
        collide2 = pygame.Rect.colliderect(ball.rect, p2.rect)

        if collide1 or collide2:
            print("collided")
            ball.speed = (ball.speed[0]*(-1), ball.speed[1])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        p.move()
        redrawWindow(win, p, p2, ball)


main()
