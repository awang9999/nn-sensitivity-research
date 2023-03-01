import numpy as np
import time
from ipycanvas import MultiCanvas, hold_canvas
from ipywidgets import Button, GridspecLayout, Output, IntSlider

import torch
from torch import nn
from MinimalSolution import MinNet

class GOLCNNInspector:
    def __init__(self, golcnn, init_state_param, c_width=512, c_height=512):
                
        self.init_state = init_state_param.copy()
        self.game_state = init_state_param.copy()
        self.field_width, self.field_height = np.shape(self.init_state)
        
        self.model = golcnn
        
        self.step_list = []
        self.compute_step_list()
        
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
    
        def reset_button_on_click(b):
            self.reset();
            
        def handle_slider_change(b):
            v = self.intslider.value
            self.change_and_render(v)
        
        self.reset_button = Button(
            description='Reset',
            disabled=False,
            tooltip='Reset',
        )
        self.reset_button.on_click(reset_button_on_click)
        
        self.intslider = IntSlider(
            value = 0,
            min = 0,
            max = self.model.num_steps,
            step = 1,
            description = 'Step:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        
        self.intslider.observe(handle_slider_change, names='value')
           
        self.ui = GridspecLayout(1,2)
        self.ui[0,0] = self.reset_button
        self.ui[0,1] = self.intslider
    
        self.render()
        
        display(self.multicanvas)
        display(self.ui)
        display(self.out)
    
    def render(self):
        with hold_canvas(self.multicanvas[1]):
            self.multicanvas[1].clear()
            self.multicanvas[1].fill_style = '#3b4252'
        
            m, n = np.shape(self.game_state)
            pixheight = self.multicanvas[1].height
            pixwidth = self.multicanvas[1].width
            m_pixels = pixheight / m
            n_pixels = pixwidth / n

            r = 0
            for row in self.game_state:
                c = 0
                for value in row:
                    if value:
                        self.multicanvas[1].fill_rect(r * m_pixels, c * n_pixels, m_pixels, n_pixels)
                    c += 1
                r += 1
                
    '''
    n must be <= the number of steps the CNN simulates
    '''
    def change_and_render(self, n):
        self.game_state = self.step_list[n]
        self.render()
    
    def reset(self):
        self.game_state = self.init_state.copy()
        self.render()
        self.intslider.value = 0
            
    def handle_mouse_event(self, x, y):
        field_x = int(np.floor(x / ( self.canvas_width / self.field_width)))
        field_y = int(np.floor(y / (self.canvas_height / self.field_height)))
        self.game_state[field_x, field_y] = (not self.game_state[field_x, field_y])
        self.compute_step_list()
        self.render()
        self.intslider.value = 0
        
    def compute_step_list(self):
        self.step_list = [self.game_state.copy()]
        for i in range(1,self.model.num_steps+1):
            
            state_prep = np.array([[self.game_state.copy()]])
            model_in = torch.from_numpy(state_prep).type(torch.float32)
            model_in = model_in.repeat(1,self.model.overcompleteness_factor,1,1)
            model_out = self.model.inspect_step(model_in, i)
            
            self.step_list.append(model_out[0][0].detach().numpy().round())