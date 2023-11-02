import pygame
import sys

# Initialize Pygame
pygame.init()

# Game screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong Game")

# Colors
WHITE = (255, 255, 255)

# Paddle dimensions and attributes
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100
PADDLE_SPEED = 2

# Create the left and right paddles
left_paddle = pygame.Rect(50, (HEIGHT - PADDLE_HEIGHT) //
                          2, PADDLE_WIDTH, PADDLE_HEIGHT)
right_paddle = pygame.Rect(WIDTH - 50 - PADDLE_WIDTH,
                           (HEIGHT - PADDLE_HEIGHT) // 2, PADDLE_WIDTH, PADDLE_HEIGHT)

# Ball dimensions and attributes
BALL_WIDTH = 10
BALL_SPEED_X = 1
BALL_SPEED_Y = 1

# Create the ball
ball = pygame.Rect((WIDTH - BALL_WIDTH) // 2, (HEIGHT -
                   BALL_WIDTH) // 2, BALL_WIDTH, BALL_WIDTH)

# Ball's initial direction
ball_direction = [1, 1]  # [x, y]

# Initialize scores
left_score = 0
right_score = 0

# Create a font for displaying scores
font = pygame.font.Font(None, 36)

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Update paddles
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and left_paddle.top > 0:
        left_paddle.y -= PADDLE_SPEED
    if keys[pygame.K_s] and left_paddle.bottom < HEIGHT:
        left_paddle.y += PADDLE_SPEED
    if keys[pygame.K_UP] and right_paddle.top > 0:
        right_paddle.y -= PADDLE_SPEED
    if keys[pygame.K_DOWN] and right_paddle.bottom < HEIGHT:
        right_paddle.y += PADDLE_SPEED

    # Move the ball
    ball.x += BALL_SPEED_X * ball_direction[0]
    ball.y += BALL_SPEED_Y * ball_direction[1]

    # Bounce the ball off the top and bottom walls
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_direction[1] *= -1

    # Check for collisions with paddles
    if ball.colliderect(left_paddle) or ball.colliderect(right_paddle):
        ball_direction[0] *= -1

    # Ball out of bounds (score)
    if ball.left <= 0:
        right_score += 1
        ball = pygame.Rect((WIDTH - BALL_WIDTH) // 2,
                           (HEIGHT - BALL_WIDTH) // 2, BALL_WIDTH, BALL_WIDTH)
        ball_direction = [1, 1]
    elif ball.right >= WIDTH:
        left_score += 1
        ball = pygame.Rect((WIDTH - BALL_WIDTH) // 2,
                           (HEIGHT - BALL_WIDTH) // 2, BALL_WIDTH, BALL_WIDTH)
        ball_direction = [-1, 1]

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw paddles and ball
    pygame.draw.rect(screen, WHITE, left_paddle)
    pygame.draw.rect(screen, WHITE, right_paddle)
    pygame.draw.ellipse(screen, WHITE, ball)

    # Draw scores
    left_text = font.render(str(left_score), True, WHITE)
    right_text = font.render(str(right_score), True, WHITE)
    screen.blit(left_text, (100, 50))
    screen.blit(right_text, (WIDTH - 100, 50))

    # Update the screen
    pygame.display.update()
