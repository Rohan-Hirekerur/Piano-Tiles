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
    def __init__(self, y, speed):
        self.pos = random.randint(0, 3)
        self.tiles = []
        self.speed = speed
        self.y = y*screen_height/4 - screen_height/4
        for i in range(0, 5):
            tile = Tile(i*screen_width/4, self.y)
            self.tiles.append(tile)
            if i == self.pos:
                self.tiles[i].is_note = True

    def move(self):
        self.y += self.speed
        for i in range(0, 4):
            self.tiles[i].y = self.y

    def display(self):
        for i in range(0, 4):
            self.tiles[i].display()

    def click(self, x):
        x = math.floor(4*x/screen_width)
        self.tiles[x].click()

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
        self.speed = 5
        for i in range(0, 5):
            row = Row(4-i, self.speed)
            print(row.y)
            self.rows.append(row)
        print("1 :", self.rows[0].y)

    def display(self):
        for row in self.rows:
            row.display()

    def move_rows(self):
        for row in self.rows:
            row.move()
        if self.rows[0].y >= screen_height:
            complete = self.rows[0].is_complete()
            #if complete:
            row = Row(0, self.speed)
            self.rows.append(row)
            self.rows.__delitem__(0)

    def click(self, pos):
        x = pos[0]
        y = pos[1]
        for i in range(0, 4):
            if y in range(int(self.rows[i].y), int(self.rows[i].y + screen_height/4)):
                self.rows[i].click(x)
        print("clicked :", x, y)

    def inc_speed(self):
        self.speed += 1
        for row in self.rows:
            row.speed = self.speed


grid = Grid()
start = False
while True:
    clock.tick()
    screen.fill(blue)
    grid.display()
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.MOUSEBUTTONUP:
            pos = pygame.mouse.get_pos()
            grid.inc_speed()
            grid.click(pos)
    grid.move_rows()
    pygame.display.update()
pygame.time.delay(2000)
pygame.quit()
