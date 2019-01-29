import pygame
import pyglet
from collections import deque
import random
import math
import numpy as np
import cnn
import tensorflow as tf


state_size = [400, 800, 3]
action_size = 5
learning_rate = 0.0002      # Alpha (aka learning rate)

# TRAINING HYPER PARAMETERS
total_episodes = 500        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 100

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyper parameters
gamma = 0.95               # Discounting rate

# MEMORY HYPER PARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

# MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = False
testing = True


possible_actions = [0, 1, 2, 3, 4]


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

_songs = ['./mp3/a1.mp3', './mp3/a1s.mp3', './mp3/b1.mp3', './mp3/c1.mp3', './mp3/c1s.mp3', './mp3/c2.mp3', './mp3/d1.mp3',
          './mp3/d1s.mp3', './mp3/e1.mp3', './mp3/f1.mp3', './mp3/f1s.mp3', './mp3/g1.mp3', './mp3/g1s.mp3']


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
        # note = random.choice(_songs)
        # pygame.mixer.music.stop()
        # pygame.mixer.music.load(note)
        # pygame.mixer.music.play()

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

    def display(self, score):
        for row in self.rows:
            row.display()
        pygame.draw.line(screen, blue, (0, 7*screen_height/8), (screen_width, 7*screen_height/8), 2)
        text = font.render(str(score), 1, red)
        text_pos = (screen_width/2 - 10, 50)
        screen.blit(text, text_pos)

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

        if event.type == pygame.KEYDOWN:
            start = True
            if event.key == pygame.K_g:
                pos = (screen_width/8, 7*screen_height/8)
                grid.click(pos)
                score += 1
                print("g")
            if event.key == pygame.K_h:
                pos = (3*screen_width/8, 7*screen_height/8)
                grid.click(pos)
                score += 1
            if event.key == pygame.K_j:
                pos = (5*screen_width/8, 7*screen_height/8)
                grid.click(pos)
                score += 1
            if event.key == pygame.K_k:
                pos = (7*screen_width/8, 7*screen_height/8)
                grid.click(pos)
                score += 1

        if score % 5 == 0:
            inc = True
    if start:
        complete = grid.move_rows(inc)
    inc = False
    grid.display(score)
    pygame.display.update()
pygame.time.delay(3000)
pygame.quit()
