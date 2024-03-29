import pygame
from enum import Enum
import torch

class Actions(Enum):
    NOTHING = 0
    RIGHT = 1
    LEFT = 2
    ROTATE_CW = 3
    ROTATE_CCW = 4


# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
RED2 = (255, 100, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 100


class Tetris:
    def __init__(self, width: int = 5, height: int = 5, ui: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define size of game board
        self.width = width
        self.height = height
        
        self.ui = ui
        if self.ui:  # setup ui properties
            pygame.init()
            self.font = pygame.font.SysFont('arial', 25)

            self.block_size = BLOCK_SIZE
            self.display = pygame.display.set_mode(
                (BLOCK_SIZE*self.width, BLOCK_SIZE*self.height)
            )
            pygame.display.set_caption('Tetris')
            self.clock = pygame.time.Clock()

        self.placedBlocks = torch.zeros((self.width, self.height), dtype=torch.int, device=self.device)
        self._reward = torch.tensor(0, dtype=torch.float, device=self.device)
        self.height_range = torch.arange(self.height, device=self.device)

        # define starting shape
        self._x = torch.tensor([[self.width//2],[self.width//2]], dtype=torch.int, device=self.device)
        self._y = torch.tensor([[self.height],[self.height-1]], dtype=torch.int, device=self.device)
        self.shape = {"x": self._x.clone(), "y": self._y.clone() }

        self.total_cleared_lines = 0
        self.game_reward = 0
        self.reset()

    @property
    def reward(self) -> torch.Tensor:
        return self._reward
    
    @reward.setter
    def reward(self, value: float):
        self._reward.fill_(value)

    def reset(self):

        # init game state
        self.done = False
        self._new_shape()
        self.placedBlocks.mul_(0)

        # init game stats
        self.score = 0
        self.game_reward = 0

    @property
    def state(self):
        state = self.placedBlocks.clone()
        state[self.x[self.shape_inview], self.y[self.shape_inview]] = 2
        return state

    def play_step(self, action: torch.Tensor):
        self.reward = 0  # reset reward at each step

        # 1. collect user input
        if self.ui:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # 2. move
        self._move(action)

        # 3. check if game over
        self._check_game_over()

        if self.ui:  # 4. update ui and clock
            self._update_ui()
            self.clock.tick(SPEED)

    def set_ui(self, value: bool):
        self.ui = value

    def _update_ui(self):
        self.display.fill(BLACK)

        for idx in self.placedBlocks.argwhere():
            x, y = idx

            pygame.draw.rect(self.display, BLUE1, pygame.Rect(x * BLOCK_SIZE, (self.height-y-1)*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(x * BLOCK_SIZE + 4, (self.height-y-1)*BLOCK_SIZE + 4, 12, 12))

        for x,y in zip(self.shape["x"], self.shape["y"]):
            pygame.draw.rect(self.display, RED, pygame.Rect(x * BLOCK_SIZE, (self.height-y-1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, RED2, pygame.Rect(x * BLOCK_SIZE + 4, (self.height-y-1) * BLOCK_SIZE + 4, 12, 12))

        x, y = self.shape["x"][0], self.shape["y"][0]
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(x * BLOCK_SIZE, (self.height-y-1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action: torch.Tensor):

        # shape points in view
        x = self.x[self.shape_inview]
        y = self.y[self.shape_inview]

        # possible actions
        if action == Actions.RIGHT.value:
            if self.x.max() < self.width-1:  # boundary check
                if not self.placedBlocks[x + 1, y].any():  # collision check
                    self.x += 1

        elif action == Actions.LEFT.value:
            if self.x.min() > 0:  # boundary check
                if not self.placedBlocks[x - 1, y].any():  # collision check
                    self.x -= 1

        elif action == Actions.ROTATE_CW.value:
            self.rotate_shape(clockwise=True)
        
        elif action == Actions.ROTATE_CCW.value:
            self.rotate_shape(clockwise=False)
        
        # update shape points in view
        x = self.x[self.shape_inview]
        y = self.y[self.shape_inview]

        # gravity
        if self.y.min() > 0:  # boundary check
            if not self.placedBlocks[x, y - 1].any():  # collision check
                self.y -= 1
            else:
                self._place_shape()
                self._clear_rows()
                self._new_shape()

        else:
            self._place_shape()
            self._clear_rows()
            self._new_shape()

    @property
    def shape_inview(self) -> torch.Tensor:
        """mask for shape points that are in view. This
        is required as shapes are initially created above the
        view and then fall into view"""
        return self.shape["y"] < self.height

    @property
    def x(self) -> torch.Tensor:
        """x index of shape points"""
        return self.shape["x"]

    @x.setter
    def x(self, value: torch.Tensor):
        self.shape["x"] = value

    @property
    def y(self) -> torch.Tensor:
        """y index of shape points"""
        return self.shape["y"]

    @y.setter
    def y(self, value: torch.Tensor):
        self.shape["y"] = value

    def _place_shape(self):

        floor = torch.zeros(len(self.state),dtype=self.state.dtype,device=self.state.device)
        floor_index = torch.argwhere(self.state == 1).to(self.state.dtype)
        floor[floor_index[:,0]] = floor_index[:,1]

        # shape points in view
        x = self.x[self.shape_inview]
        y = self.y[self.shape_inview]

        # place shape
        self.placedBlocks[x, y] = 1

        # reward for filling out board horizontally
        
        self.reward += -1 if torch.any(self.state[self.x,self.y-1] == 0) else 1
        self.game_reward += self.reward

    def _clear_rows(self):
        isfull = self.placedBlocks.all(axis=0)
        num_full = sum(isfull)

        updatedBlocks = self.placedBlocks[:, ~isfull]

        self.placedBlocks.fill_(value=0)
        self.placedBlocks[:, :self.height-num_full] = updatedBlocks

        self.score += num_full
        self.total_cleared_lines += num_full

        # reward for clearing rows
        # self.reward += 10*num_full
  
    def _new_shape(self):
        """Create a new shape above the view"""
        self.shape["x"] = self._x.clone()
        self.shape["y"] = self._y.clone()

        return self.shape

    def _check_game_over(self):
        """check if any of the placed blocks hit the top of the screen"""
        self.done = self.placedBlocks[:, -1].any()
        #reward for game over
        self.reward -= 10 if self.done else 0
        return self.done

    def rotate_shape(self, clockwise: bool = True):
        """Rotate shape clockwise"""

        x_median = torch.unique(self.x.float()).mean()
        y_median = torch.unique(self.y.float()).mean()

        if clockwise == False:
            # rotate shape
            x_rotated = x_median - (self.y - y_median)
            y_rotated = y_median + (self.x - x_median)
        else:
            # rotate shape
            x_rotated = x_median + (self.y - y_median)
            y_rotated = y_median - (self.x - x_median)

        if x_rotated.max() < self.width-1 and x_rotated.min() > 0:  # boundary check
            # check for collisions
            x_rotated = x_rotated.int()
            y_rotated = y_rotated.int()
            if not self.placedBlocks[x_rotated[y_rotated < self.height], y_rotated[y_rotated < self.height]].any():

                self.x = x_rotated
                self.y = y_rotated

    
    def _check_move(self,action):
    # shape points in view
        x = self.x[self.shape_inview]
        y = self.y[self.shape_inview]

        new_x = x.clone()
        new_y = y.clone()

        # possible actions
        if action.item() == Actions.RIGHT.value:
            if new_x.max() < self.width-1:  # boundary check
                if not self.placedBlocks[x + 1, y].any():  # collision check
                    new_x += 1

        elif action.item() == Actions.LEFT.value:
            if new_x.min() > 0:  # boundary check
                if not self.placedBlocks[x - 1, y].any():  # collision check
                    new_x -= 1

        elif action.item() == Actions.ROTATE_CW.value:
            new_x, new_y = self.rotate_xy(x, y, clockwise=True)
        
        elif action.item() == Actions.ROTATE_CCW.value:
            new_x, new_y = self.rotate_xy(x, y, clockwise=False)
        
        # update shape points in view
        x = new_x
        y = new_y

        unique_x = torch.unique(x)
        unique_y = torch.zeros_like(unique_x)
        for u in range(len(unique_x)):
            unique_y[u] =y[torch.where(x==unique_x[u])[0]].min().unsqueeze(0)

        # find highest y index where state[x,y] == 1 for each unique x
        y_max = torch.zeros_like(unique_y)
        for i, xs in enumerate(unique_x):
            if (self.state[xs]==1).any():
                y_max[i] = torch.where(self.state[xs]==1)[0].max()+1
            else:
                y_max[i] = 0
        diff = unique_y-y_max
        _, counts = torch.unique(diff, return_counts = True)
        score = counts.max()/len(y_max) #+ 2*(self.height - y_max.max())/self.height
        return score
    
    def rotate_xy(self, x, y, clockwise: bool = True):
        """Rotate shape clockwise"""

        x_median = torch.unique(x.float()).mean()
        y_median = torch.unique(y.float()).mean()

        if clockwise == False:
            # rotate shape
            x_rotated = x_median - (y - y_median)
            y_rotated = y_median + (x - x_median)
        else:
            # rotate shape
            x_rotated = x_median + (y - y_median)
            y_rotated = y_median - (x - x_median)

        if x_rotated.max() < self.width-1 and x_rotated.min() > 0:  # boundary check
            # check for collisions
            x_rotated = x_rotated.int()
            y_rotated = y_rotated.int()
            if not self.placedBlocks[x_rotated[y_rotated < self.height], y_rotated[y_rotated < self.height]].any():

                x = x_rotated
                y = y_rotated
        return x,y