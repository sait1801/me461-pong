import pygame
import random

WIDTH = 1000
HEIGHT = 500


class Ball:
    def __init__(self, posx, posy, speed, color):
        self.posx = posx
        self.posy = posy
        self.radius = 10
        self.speed = (5, 5)  # in terms of (vecx, vecy )
        self.color = color
        self.rect = pygame.Rect(
            posx, posy, self.radius, self.radius)

        self.centre = (posx, posy)

    def draw(self, win):

        pygame.draw.circle(win, self.color, self.centre, self.radius)
        # pygame.draw.rect(win, self.color, self.rect)

    def move(self):
        # print(self.speed)

        self.posx += self.speed[0]
        self.posy += self.speed[1]

        if self.posy >= HEIGHT or self.posy <= 0:
            self.speed = (self.speed[0], self.speed[1]*(-1))
            self.posy += self.speed[1]

        if self.posx <= 0 or self.posx >= WIDTH:
            # player_score += 1
            self.speed = (self.speed[0]*(-1), self.speed[1])
            self.posx += self.speed[0]
            # Check for collisions with paddles

        self.update()

    def update(self):
        # print(f"{self.posx}, {self.posy}")
        self.centre = (int(self.posx), int(self.posy))
        self.rect = pygame.Rect(self.posx, self.posy,
                                self.radius, self.radius)
