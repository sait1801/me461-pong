# Player class
class Striker:

    # Take the initial position,
    # dimensions, speed and color of the object
    def __init__(self, posx, posy, width, height, speed, color):
        self.posx = posx
        self.posy = posy
        self.width = width
        self.height = height
        self.speed = speed
        self.color = color
        # Rect that is used to control the
        # position and collision of the object
        self.geekRect = pygame.Rect(posx, posy, width, height)
        # Object that is blit on the screen
        self.geek = pygame.draw.rect(screen, self.color, self.geekRect)

    # Used to display the object on the screen
    def display(self):
        self.geek = pygame.draw.rect(screen, self.color, self.geekRect)

    # Used to update the state of the object
    # yFac represents the direction of the striker movement
    # if yFac == -1 ==> The object is moving upwards
    # if yFac == 1 ==> The object is moving downwards
    # if yFac == 0 ==> The object is not moving
    def update(self, yFac):
        self.posy = self.posy + self.speed*yFac

        # Restricting the striker to be below
        # the top surface of the screen
        if self.posy <= 0:
            self.posy = 0
        # Restricting the striker to be above
        # the bottom surface of the screen
        elif self.posy + self.height >= HEIGHT:
            self.posy = HEIGHT-self.height

        # Updating the rect with the new values
        self.geekRect = (self.posx, self.posy, self.width, self.height)

    # Used to render the score on to the screen
    # First, create a text object using the font.render() method
    # Then, get the rect of that text using the get_rect() method
    # Finally blit the text on to the screen
    def displayScore(self, text, score, x, y, color):
        text = font20.render(text+str(score), True, color)
        textRect = text.get_rect()
        textRect.center = (x, y)

        screen.blit(text, textRect)

    def getRect(self):
        return self.geekRect
