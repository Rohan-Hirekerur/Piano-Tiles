import pygame
import pyglet
from collections import deque

white = (255, 255, 255)
red = (255, 0, 0)
black = (0, 0, 0)

pygame.init()
screen_width =
screen_height = 500
size = screen_width, screen_height
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Piano tiles")
pygame.display.update()
clock = pyglet.clock.Clock()
clock.set_fps_limit(60)
font = pygame.font.Font(None, 50)


class Tile:
    def __init__(self, is_note):
        self.is_note = is_note
        self.width = screen_width/4
        self.height = screen_height/4
        self.is_clicked = False

    def click(self):
        self.is_clicked = True


class Grid:
    def __init__(self):
        self.height = 4
        self.width = 4
        self.queue = 


