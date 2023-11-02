import pygame


WIDTH = 1000
HEIGHT = 500


class Player():
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.rect = pygame.Rect(x, y, width, height)
        self.vel = 3

    def draw(self, win):
        pygame.draw.rect(win, self.color, self.rect)

    def move(self):
        keys = pygame.key.get_pressed()

        # if keys[pygame.K_LEFT]:
        #     self.x -= self.vel

        # if keys[pygame.K_RIGHT]:
        #     self.x += self.vel

        if keys[pygame.K_UP]:
            self.y -= self.vel

        if keys[pygame.K_DOWN]:
            self.y += self.vel

        if self.y + self.height >= HEIGHT:
            self.y -= self.y + - self.height
        if self.y <= 0:
            self.y -= self.y

        self.update()

    def update(self):
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
