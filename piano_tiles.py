import pygame
import pyglet
from collections import deque
import random

white = (255, 255, 255)
blue = (0, 0, 255)
black = (0, 0, 0)

pygame.init()
screen_width = 400
screen_height = 800
size = screen_width, screen_height
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Piano tiles")
pygame.display.update()
clock = pyglet.clock.Clock()
clock.set_fps_limit(60)
font = pygame.font.Font(None, 50)


class Tile:
    def __init__(self, x, y):
        self.is_note = False
        self.width = screen_width/4
        self.height = screen_height/4
        self.is_clicked = False
        self.x = x
        self.y = y
        print(self.x, self.y)

    def click(self):
        self.is_clicked = True

    def display(self):
        if not self.is_note:
            pygame.draw.rect(screen, white, [self.x, self.y, self.width, self.height])
        else:
            pygame.draw.rect(screen, black, [self.x, self.y, self.width, self.height])


class Row:
    def __init__(self, y):
        self.pos = random.randint(0, 3)
        self.tiles = []
        self.y = y*screen_height/4
        print("pos :", self.pos)
        for i in range(0, 4):
            tile = Tile(i*screen_width/4, self.y)
            self.tiles.append(tile)
            if i == self.pos:
                self.tiles[i].is_note = True

    def move(self):
        self.y += 10
        for i in range(0, 4):
            self.tiles[i].y = self.y

    def display(self):
        for i in range(0, 4):
            self.tiles[i].display()

    def is_complete(self):
        for i in range(0, 4):
            if self.tiles[i].is_note and not self.tiles[i].is_clicked:
                return False
            if not self.tiles[i].is_note and self.tiles[i].is_clicked:
                return False
        return True


class Grid:
    def __init__(self):
        self.rows = deque()
        for i in range(0, 4):
            row = Row(3-i)
            self.rows.append(row)

    def display(self):
        for row in self.rows:
            row.display()


grid = Grid()
while True:
    clock.tick()
    screen.fill(blue)
    grid.display()
    pygame.display.update()
pygame.quit()
