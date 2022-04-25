import numpy as np
import matplotlib.pyplot as plt


class Checker:
    
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = self.draw()
    
    def draw(self):
        repeat = self.resolution // (2* self.tile_size)

        black = np.zeros((self.tile_size, self.tile_size))
        white = np.ones((self.tile_size, self.tile_size))

        a1 = np.hstack([black, white])
        a2 = np.hstack([white, black])
        a = np.concatenate([a1, a2])

        output = np.tile(a, (repeat, repeat))
        
        return output
        
    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()
        

class Circle:
    
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position =position
        self.output = self.draw()
        
    def draw(self):
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        xx, yy = np.meshgrid(x, y)
        
        circle = (xx - self.position[0])**2 + (yy - self.position[1])**2
        output = (circle <= (self.radius)**2)
        return output
        
    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()
        
        
class Spectrum:
    
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = self.draw()
        
    def draw(self):
        
        arr1 = np.arange(0, 1, 1/self.resolution)
        arr2 = np.arange(1, 0, -1/self.resolution)

        br1, br2= np.meshgrid(arr1,arr1)
        bottom_right= br1*br2

        tl1, tl2= np.meshgrid(arr2,arr2)
        top_left = tl1*tl2
        
        tr1, tr2 = np.meshgrid(arr1, arr2)
        top_right= tr1*tr2

        bl1, bl2 = np.meshgrid(arr2, arr1)
        bottom_left= bl1*bl2

        bottom=br2

        rgb = np.dstack((top_right+bottom_right, bottom , top_left+bottom_left)) 
        output= rgb
        return output
        
    def show(self):
        plt.imshow(self.output)
        plt.show()

