import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Actions(Enum):
    RIGHT = 1
    LEFT = 2
    ROTATE = 3
    NOTHING = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 50

class TetrisAI:

    def __init__(self, w=120, h=600):
        self.w = w
        self.h = h
        self.block_size = BLOCK_SIZE
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Tetris')
        self.clock = pygame.time.Clock()
        self.reset()
        self.n_cleared_lines_total = 0
        self.n_cleared_lines = 0
        self.reward = 0
        


    def reset(self):
        # init game state

        self.centerPoint = Point(self.w/2, self.h - BLOCK_SIZE)
        self.shape = [self.centerPoint, Point(self.centerPoint.x+BLOCK_SIZE, self.centerPoint.y), Point(self.centerPoint.x-BLOCK_SIZE, self.centerPoint.y), Point(self.centerPoint.x, self.centerPoint.y+BLOCK_SIZE)]

        self.score = 0
        self.placedBlocks = np.zeros((self.w//BLOCK_SIZE, self.h//BLOCK_SIZE))
        self.frame_iteration = 0


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        
        # 3. check if game over
        if self.reward != 0:
            reward = self.reward
            self.reward = 0
        else:
            reward = 0
        game_over = False
        reward += 10*self.n_cleared_lines
        self.n_cleared_lines = 0

        if self._check_game_over():
            game_over = True
            reward -= 1
            
            return reward, game_over, self.score
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def _update_ui(self):
        self.display.fill(BLACK)

        for row in range(len(self.placedBlocks)):
            for col in range(len(self.placedBlocks[row])):
                if self.placedBlocks[row][col] == 1:
                    pygame.draw.rect(self.display, BLUE1, pygame.Rect((row)*BLOCK_SIZE, self.h-(col+1)*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.display, BLUE2, pygame.Rect((row)*BLOCK_SIZE+4, self.h-(col+1)*BLOCK_SIZE+4, 12, 12))


        for point in self.shape:
            pygame.draw.rect(self.display, RED, pygame.Rect(point.x, self.h-point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(point.x+4, self.h-point.y+4, 12, 12))

        pygame.draw.rect(self.display, BLUE1, pygame.Rect(self.centerPoint.x, self.h-self.centerPoint.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [right, left, rotate, nothing]
        shape = self.shape.copy()
        allowed = True

        if np.array_equal(action, [1, 0, 0, 0]):

            for p, point in enumerate(self.shape):
                x = point.x + BLOCK_SIZE
                y = point.y
                try:
                    if self.placedBlocks[int(x//BLOCK_SIZE), int(y//BLOCK_SIZE)-1] == 1:
                        shape = self.shape
                        allowed = False
                        break
                    elif x < 0 or y-BLOCK_SIZE == 0:
                        shape = self.shape
                        allowed = False
                        break
                except Exception as e:
                    shape = self.shape
                    allowed = False
                    break

                shape[p] = Point(x, y)
            if allowed:
                    self.centerPoint = Point(self.centerPoint.x+BLOCK_SIZE, self.centerPoint.y)
                
        elif np.array_equal(action, [0, 1, 0, 0]):


            for p, point in enumerate(self.shape):
                x = point.x - BLOCK_SIZE
                y = point.y
                try:
                    if self.placedBlocks[int(x//BLOCK_SIZE), int(y//BLOCK_SIZE)-1] == 1:
                        shape = self.shape
                        allowed = False
                        break
                    elif x < 0 or y-BLOCK_SIZE == 0:
                        shape = self.shape
                        allowed = False
                        break
                except Exception as e:
                    shape = self.shape
                    allowed = False
                    break
                shape[p] = Point(x, y)
            if allowed:
                self.centerPoint = Point(self.centerPoint.x-BLOCK_SIZE, self.centerPoint.y)

        elif np.array_equal(action, [0, 0, 1, 0]):
            shape = self.rotate_shape(self.shape, (self.centerPoint.x, self.centerPoint.y))
            for p, point in enumerate(shape):
                x = point.x
                y = point.y
                try:
                    if self.placedBlocks[int(x//BLOCK_SIZE), int(y//BLOCK_SIZE)-1] == 1:
                        shape = self.shape
                        allowed = False
                        break
                    elif x < 0 or y-BLOCK_SIZE == 0:
                        shape = self.shape
                        allowed = False
                        break
                except Exception as e:
                    shape = self.shape
                    allowed = False
                    break
                shape[p] = Point(x, y)

        for p,point in enumerate(shape):
            x = point.x
            y = point.y - BLOCK_SIZE
            shape[p] = Point(x, y)

        self.centerPoint = Point(self.centerPoint.x, self.centerPoint.y-BLOCK_SIZE)
        self.shape = shape

        if self._check_bottom():
            self._place_shape()
            self._clear_rows()
            self._new_shape()
            

    def _is_valid(self):
        for point in self.shape:
            if point.x > self.w - BLOCK_SIZE or point.x < 0:
                return False
        return True


    def _place_shape(self):
        self.reward = self.get_reward()
        for point in self.shape:
            self.placedBlocks[int(point.x//BLOCK_SIZE), int(point.y//BLOCK_SIZE)] = 1

    
    def _clear_rows(self):
        cleared = False
        for row in range(self.h//BLOCK_SIZE):
            if np.all(self.placedBlocks[:,row] == 1):
                print(row)
                self.placedBlocks[:,row:-2] = self.placedBlocks[:,row+1:-1]
                self.score += 1
                self.n_cleared_lines_total += 1
                self.n_cleared_lines += 1
                cleared = True
        # if not cleared:
        #     self.n_cleared_lines = 0

                
    def _new_shape(self):
        centerX = random.randint(1, self.w//BLOCK_SIZE-2)
        self.centerPoint = Point(centerX*BLOCK_SIZE, self.h - BLOCK_SIZE)
        self.shape = [self.centerPoint, Point(self.centerPoint.x+BLOCK_SIZE, self.centerPoint.y), Point(self.centerPoint.x-BLOCK_SIZE, self.centerPoint.y), Point(self.centerPoint.x, self.centerPoint.y+BLOCK_SIZE)]
        rotation_num = random.randint(0,3)
        for i in range(rotation_num):
            self.shape = self.rotate_shape(self.shape, (self.centerPoint.x, self.centerPoint.y))

    def _check_game_over(self):
        #check if highest point of shape block is below the top of the screen

        for place in self.placedBlocks.T[-2]:
            if place == 1:
                return True
    #check if the lowest point of the shape is adjacent to a placed block or the bottom of the screen
    def _check_bottom(self):
        
        for point in self.shape:
            if point.y == 0 or self.placedBlocks[int(point.x//BLOCK_SIZE), int(point.y//BLOCK_SIZE)-1] == 1:
                return True
        return False

    def rotate_shape(self,shape, center_point):
        # Translate the shape so that the center point is at the origin
        translated_shape = [(x - center_point[0], y - center_point[1]) for x, y in shape]

        # Rotate each point in the translated shape around the origin (0, 0)
        rotated_shape = [(round(x * math.cos(math.pi / 2) - y * math.sin(math.pi / 2)),
                        round(x * math.sin(math.pi / 2) + y * math.cos(math.pi / 2)))
                        for x, y in translated_shape]

        # Translate the shape back to its original position
        final_rotated_shape = [Point(x + center_point[0], y + center_point[1]) for x, y in rotated_shape]

        return final_rotated_shape
    
    def get_reward(self):
        #find highest point of in placed blocks
        highest = 0
        for col in self.placedBlocks:
            #find location of highest block in column
            ind = np.where(col == 1)[0]
            if len(ind) == 0:
                ind = 0
            else:
                ind = ind[-1]
            if ind > highest:
                highest = ind
        
        lowest = self.h//BLOCK_SIZE
        #find lowest point of shape
        for point in self.shape:
            if point.y//BLOCK_SIZE < lowest:
                lowest = point.y//BLOCK_SIZE
        
        return highest - lowest