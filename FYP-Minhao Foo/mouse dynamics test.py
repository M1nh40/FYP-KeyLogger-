import time
from datetime import datetime
import csv
import pandas as pd
import math
import numpy as np
from pynput.mouse import Listener
import threading

df = pd.DataFrame(columns=['x','y','time', 'Distance', 'velocity'])
velotime = []
clicked = threading.Event()
start=time.time()
def time_convert(sec):
    sec = sec % 60
    print("time lapsed = {0}".format(sec))



def on_move(x, y):
    global df
    last = start
    vtime =time.time() - last
    print(x, y)
    distance = math.sqrt((y**2) + (x**2))
    print(distance, vtime)
    velo=distance/vtime
    last = time.time()              #issue: cannot seem to make the correct format for timing
    df=df.append({'x': x, 'y': y,'time':vtime, 'Distance': distance, 'velocity':velo}, ignore_index = True)


def on_click(x, y, button, pressed):
    print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
    clicked.set()
    """if not pressed:
        # Stop listener
        return False"""
    return False



with Listener(on_move = on_move,on_click = on_click ) as listener:
    listener.join()
try:
    listener.wait()
    clicked.wait()
finally:
    listener.stop()


meandis=df['Distance'].mean()
meanvel=df['velocity'].mean()
df=df.append({'Avg Dist':meandis, 'Avg velo': meanvel}, ignore_index=True)
df.to_csv(f'D:/source/repos/Keystroke UI/mousetest.csv', index=False)

