import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Annotate(object):
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.rect = Rectangle((0,0), 1, 1)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release_draw(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.fig.canvas.draw()
        time.sleep(0.3)
        plt.close()

    def select_bb(self, image, message):
        print(f'{message}')
        self.ax.add_patch(self.rect)
        cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release_draw)
        self.ax.imshow(image)
        plt.show(block=True)
        self.fig.canvas.mpl_disconnect(cid_press)
        self.fig.canvas.mpl_disconnect(cid_release)
        return np.array([self.x0, self.y0, self.x1, self.y1])

    def select_pt(self, image, message):
        print(f'{message}')
        cid = self.fig.canvas.mpl_connect('button_release_event', self.on_press)
        self.ax.imshow(image)
        plt.show(block=True)
        self.fig.canvas.mpl_disconnect(cid)
        plt.close()
        return np.array([self.x0, self.y0])