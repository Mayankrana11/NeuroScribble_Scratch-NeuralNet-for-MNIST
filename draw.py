import pygame
import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter, zoom
from model import NeuralNet
from utils import load


SIZE = 28
SCALE = 20
WINDOW = SIZE * SCALE

BRUSH_RADIUS = 2.2
BRUSH_STRENGTH = 0.30


pygame.init()
screen = pygame.display.set_mode((WINDOW, WINDOW))
pygame.display.set_caption("MNIST Draw (Enter = Predict, C = Clear)")
clock = pygame.time.Clock()

# Load trained model
model = NeuralNet()
load(model)

grid = np.zeros((SIZE, SIZE), dtype=np.float32)

drawing = False
running = True


def draw_grid():
    for y in range(SIZE):
        for x in range(SIZE):
            v = int(grid[y, x] * 255)
            pygame.draw.rect(
                screen,
                (v, v, v),
                (x * SCALE, y * SCALE, SCALE, SCALE)
            )


def apply_brush(cx, cy):
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            x = cx + dx
            y = cy + dy
            if 0 <= x < SIZE and 0 <= y < SIZE:
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= BRUSH_RADIUS:
                    value = (1 - dist / BRUSH_RADIUS) * BRUSH_STRENGTH
                    grid[y, x] = min(1.0, grid[y, x] + value)


def preprocess(img):
    #Remove weak noise
    img = np.where(img > 0.05, img, 0)

    if img.sum() == 0:
        return img

    #Bounding box
    ys, xs = np.nonzero(img)
    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()
    digit = img[top:bottom + 1, left:right + 1]

    #Resize longest side to 20 pixels
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    digit = zoom(digit, scale, order=1)

    #Pad to 28x28
    canvas = np.zeros((28, 28), dtype=np.float32)
    h, w = digit.shape
    y_off = (28 - h) // 2
    x_off = (28 - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = digit

    # Center by center-of-mass
    cy, cx = center_of_mass(canvas)
    if not np.isnan(cx):
        canvas = np.roll(canvas, int(14 - cy), axis=0)
        canvas = np.roll(canvas, int(14 - cx), axis=1)

    #Slight blur (MNIST-like)
    canvas = gaussian_filter(canvas, sigma=0.7)

    #Normalize
    if canvas.max() > 0:
        canvas /= canvas.max()

    return canvas


while running:
    screen.fill((0, 0, 0))
    draw_grid()
    pygame.display.flip()
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                grid[:] = 0
                print("Canvas cleared")

            elif event.key == pygame.K_RETURN:
                processed = preprocess(grid)
                x = processed.reshape(1, 784)
                probs = model.forward(x)[0]
                pred = probs.argmax()

                print("\nPrediction:", pred)
                for i, p in enumerate(probs):
                    print(f"{i}: {p:.4f}")

    if drawing:
        mx, my = pygame.mouse.get_pos()
        cx = mx // SCALE
        cy = my // SCALE
        if 0 <= cx < SIZE and 0 <= cy < SIZE:
            apply_brush(cx, cy)

pygame.quit()
