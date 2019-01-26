import pygame
import pyglet
from collections import deque
import random
import math

white = (255, 255, 255)
blue = (0, 0, 255)
black = (0, 0, 0)
red = (255, 0, 0)

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

    def click(self):
        self.is_clicked = True

    def display(self):
        if not self.is_note:
            if not self.is_clicked:
                pygame.draw.rect(screen, white, [self.x, self.y, self.width, self.height])
            else:
                pygame.draw.rect(screen, red, [self.x, self.y, self.width, self.height])
        else:
            if not self.is_clicked:
                pygame.draw.rect(screen, black, [self.x, self.y, self.width, self.height])
            else:
                pygame.draw.rect(screen, white, [self.x, self.y, self.width, self.height])


class Row:
    def __init__(self, y):
        self.pos = random.randint(0, 3)
        self.tiles = []
        self.y = y
        for i in range(0, 4):
            tile = Tile(i*screen_width/4, self.y)
            self.tiles.append(tile)
            if i == self.pos:
                self.tiles[i].is_note = True

    def move(self, speed):
        self.y += speed
        for tile in self.tiles:
            tile.y = self.y

    def display(self):
        for tile in self.tiles:
            tile.display()

    def click(self, x):
        x = math.floor(4*x/screen_width)
        self.tiles[x].click()

    def is_complete(self):
        for i in range(0, 4):
            if self.tiles[i].is_note and not self.tiles[i].is_clicked:
                return False
        return True

    def false_clicked(self):
        for i in range(0, 4):
            if not self.tiles[i].is_note and self.tiles[i].is_clicked:
                return True
        return False


class Grid:
    def __init__(self):
        self.rows = deque()
        self.speed = 5
        for i in range(0, 5):
            y = (4-i) * screen_height / 4 - screen_height / 4
            row = Row(y)
            print(row.y)
            self.rows.append(row)
        print("1 :", self.rows[0].y)

    def display(self):
        for row in self.rows:
            row.display()

    def move_rows(self, inc):
        for row in self.rows:
            if row.false_clicked():
                return False
        if self.rows[0].y >= screen_height:
            complete = self.rows[0].is_complete()
            if complete:
                y = self.rows[4].y - screen_height / 4
                row = Row(y)
                self.rows.__delitem__(0)
                self.rows.append(row)
            return complete

        if inc and self.speed < 15:
            self.speed *= 1.05
        for row in self.rows:
                row.speed = self.speed
        for row in self.rows:
            row.move(self.speed)
        return True

    def click(self, pos):
        x = pos[0]
        y = pos[1]
        for row in self.rows:
            if y >= row.y:
                row.click(x)
                break
        print("clicked :", x, y)


grid = Grid()
start = False
inc = False
complete = True
score = 0
while complete:
    clock.tick()
    screen.fill(blue)
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            start = True
            pos = pygame.mouse.get_pos()
            grid.click(pos)
            score += 1
            if score % 5 == 0:
                inc = True
    if start:
        complete = grid.move_rows(inc)
    inc = False
    grid.display()
    pygame.display.update()
pygame.time.delay(2000)
pygame.quit()
