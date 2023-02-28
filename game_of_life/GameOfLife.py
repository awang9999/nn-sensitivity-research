import numpy as np
import time
from ipycanvas import MultiCanvas, hold_canvas
from ipywidgets import Button, GridspecLayout, Output

import torch
from torch import nn
from MinimalSolution import MinNet

class GameOfLifeEngine:
    '''
    Initializes core parameters of a GOL simulation. 
    That is, the initial state, game state, and field width/height.
    '''
    def __init__(self, init_state_param):
        self.init_state = init_state_param.copy()
        self.game_state = init_state_param.copy()
        self.field_width, self.field_height = np.shape(self.init_state)
        
    '''
    step() is an 'abstract' function that each concrete GOL Engine must implement. 
    Its purpose is to simulate 1 step of Conway's game of life.
    '''
    def step(self):
        pass
    
    def step_n(self, n):
        for i in range(n):
            self.step()
    
    '''
    The reset() function simply sets the game state back to the initial state.
    '''
    def reset(self):
        self.game_state = self.init_state.copy()

'''
StandardEngine 'implements' the GameOfLifeEngine interface by translating
the exact rules from Wikipedia to code.
'''
class StandardEngine(GameOfLifeEngine):
    def step(self):
        m, n = self.field_width, self.field_height
        next_state = np.zeros((m, n), dtype=int)
        for i in range(m):
            for j in range(n):
                live_neighbors = 0
            
                live_neighbors += self.game_state[i-1,j-1]
                live_neighbors += self.game_state[i-1,j]
                live_neighbors += self.game_state[i-1,j+1 if j+1 < n else 0]
            
                live_neighbors += self.game_state[i,j-1]
                live_neighbors += self.game_state[i,j+1 if j+1 < n else 0]
            
                live_neighbors += self.game_state[i+1 if i+1 < m else 0, j-1]
                live_neighbors += self.game_state[i+1 if i+1 < m else 0, j]
                live_neighbors += self.game_state[i+1 if i+1 < m else 0, j+1 if j+1 < n else 0]
            
                if live_neighbors < 2: # dies by underpopulation (rule 1)
                    next_state[i,j] = 0 
                elif live_neighbors > 3: # dies by overpopulation (rule 3)
                    next_state[i,j] = 0 
                elif self.game_state[i,j] == 0 and live_neighbors == 3: # reproduction (rule 4)
                    next_state[i,j] = 1
                elif self.game_state[i,j] == 1 and (live_neighbors == 2 or live_neighbors == 3): # (rule 2)
                    next_state[i,j] = 1
        self.game_state = next_state

'''
ConvEngine 'implements' the GameOfLifeEngine interface using a 
Convolutional Neural Network.
'''
class ConvEngine(GameOfLifeEngine):
    def __init__(self, init_state_param, n_steps=1):
        super(ConvEngine, self).__init__(init_state_param)
        self.model = MinNet(n_steps=n_steps)
        self.model.eval()
    
    def step(self):
        state_prep = np.array([[self.game_state]])
        model_in = torch.from_numpy(state_prep).type(torch.float32)
        model_out = self.model.forward(model_in)
        next_state = model_out[0][0].numpy().round()
        self.game_state = next_state

class GameOfLife:
    def __init__(self, gol_engine, c_width=512, c_height=512):
        # Connect engine
        self.engine = gol_engine
        
        # Set up debug output
        self.out = Output()
        
        # Set up canvas
        self.canvas_width = c_width
        self.canvas_height = c_height
        self.multicanvas = MultiCanvas(2, width=self.canvas_width, height=self.canvas_height)
        
        self.multicanvas[0].fill_style = '#eceff4'
        self.multicanvas[0].fill_rect(0, 0, self.multicanvas.width, self.multicanvas.height)
        
        @self.out.capture()
        def handle_mouse_up(x, y):
            self.handle_mouse_event(x, y)

        self.multicanvas.on_mouse_up(handle_mouse_up)
        
        def next_button_on_click(b):
            self.step_and_render();
    
        def reset_button_on_click(b):
            self.reset();
            
        def animation_button_on_click(b):
            self.play_animation_100_steps();
        
        self.next_button = Button(
            description='Next Step',
            disabled=False,
            tooltip='Next Step',   
        )
        self.next_button.on_click(next_button_on_click)
        
        self.reset_button = Button(
            description='Reset',
            disabled=False,
            tooltip='Reset',
        )
        self.reset_button.on_click(reset_button_on_click)
        
        self.animation_button = Button(
            description='Step 100x',
            disabled=False,
            tooltip='Play animation to step 100x'
        )
        self.animation_button.on_click(animation_button_on_click)
           
        self.ui = GridspecLayout(1,3)
        self.ui[0,0] = self.reset_button
        self.ui[0,1] = self.next_button
        self.ui[0,2] = self.animation_button
    
        self.render()
        
        display(self.multicanvas)
        display(self.ui)
        display(self.out)
    
    def render(self):
        with hold_canvas(self.multicanvas[1]):
            self.multicanvas[1].clear()
            self.multicanvas[1].fill_style = '#3b4252'
        
            m, n = np.shape(self.engine.game_state)
            pixheight = self.multicanvas[1].height
            pixwidth = self.multicanvas[1].width
            m_pixels = pixheight / m
            n_pixels = pixwidth / n

            r = 0
            for row in self.engine.game_state:
                c = 0
                for value in row:
                    if value:
                        self.multicanvas[1].fill_rect(r * m_pixels, c * n_pixels, m_pixels, n_pixels)
                    c += 1
                r += 1
                
    def step_and_render(self):
        self.engine.step()
        self.render()
    
    def reset(self):
        self.engine.reset()
        self.render()
        
    def play_animation_100_steps(self):
        for i in range(100):
            self.step_and_render()
            time.sleep(0.04)
            
    def handle_mouse_event(self, x, y):
        field_x = int(np.floor(x / ( self.canvas_width / self.engine.field_width)))
        field_y = int(np.floor(y / (self.canvas_height / self.engine.field_height)))
        self.engine.game_state[field_x, field_y] = (not self.engine.game_state[field_x, field_y])
        self.render()