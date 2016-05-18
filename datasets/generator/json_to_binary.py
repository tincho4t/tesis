from os import listdir
import json
import numpy as np
from array import *

import copy
import gc
import rlcompleter, readline
from PIL import Image
readline.parse_and_bind('tab:complete')

def load_dataset(directory, size=(32, 32)):
    ticks = np.empty((0,), dtype=np.integer)
    seconds = np.empty((0,), dtype=np.integer)
    miliseconds = np.empty((0,), dtype=np.integer)
    goal_as = np.empty((0,), dtype=np.integer)
    goal_bs = np.empty((0,), dtype=np.integer)
    screens = np.empty((0,) + size + (3,))
    for f in listdir(directory):
        with open(directory+"/"+f) as data_file:    
            data = json.load(data_file)
            tick, second, milisecond, goal_a, goal_b, screen = process_file(data)
            ticks = np.concatenate((ticks, tick), axis=0)
            seconds = np.concatenate((seconds, second), axis=0)
            miliseconds = np.concatenate((miliseconds, milisecond), axis=0)
            goal_as = np.concatenate((goal_as, goal_a), axis=0)
            goal_bs = np.concatenate((goal_bs, goal_b), axis=0)
            screens = np.concatenate((screens, screen), axis=0)
    return ticks, seconds, miliseconds, goal_as, goal_bs, screens 


def process_file(data, size=(32, 32)):
    n = len(data)
    tick = np.zeros((n,), dtype=np.integer)
    second = np.zeros((n,), dtype=np.integer)
    milisecond = np.zeros((n,), dtype=np.integer)
    goal_a = np.zeros((n,), dtype=np.integer)
    goal_b = np.zeros((n,), dtype=np.integer)
    screen_array_size = (n,) + size + (3,)
    screen = np.empty(screen_array_size)
    for i in range(len(data)):
        milisecond[i] = data[i]['miliseconds']
        second[i] = data[i]['seconds']
        goal_a[i] = data[i]['goal_team_a']
        goal_b[i] = data[i]['goal_team_b']
        tick[i] = data[i]['tick']
        screen[i] = np.array(Image.fromarray(np.array(data[i]['screen']).astype(np.uint8),'RGB').resize(size))
    return tick, second, milisecond, goal_a, goal_b, screen


def mark_pre_goal(goals, seconds_in_future=2):
    res = np.zeros_like(goals)
    n = len(goals)
    i = 0
    while i < n:
        to = min(n-1, i+seconds_in_future)
        if goals[to]==1:
            # Mark from here till goal as 1s
            res[i:(to+1)]=1
            # We only mark the goal once, and then skip 10 seconds of celebrations
            i+=10
            continue
        i+=1
    return(res)


def store_dataset_for_tf_as_cifar(goals, screens, directory, new_every_n_lines=160, size=(32, 32)):
    n = len(goals)
    j = 0
    while j < n:
        image_bytes = size[0]*size[1]*3
        label_bytes = 1
        out = np.empty((new_every_n_lines, label_bytes+image_bytes), np.uint8)
        data = array('B')
        print(j)
        for i in range(j,min(n, j+new_every_n_lines)):
            data.append(goals[i])
            for color in range(0,3):
                for x in range(0,size[0]):
                    for y in range(0,size[1]):
                        data.append(int(screens[i][x,y][color]))
        output_file = open(directory+"/out_big_"+str(int(j/new_every_n_lines))+".bin", 'wb')
        data.tofile(output_file)
        output_file.close()
        j += new_every_n_lines



ticks, seconds, miliseconds, goal_as, goal_bs, screens = load_dataset("/home/eitan/Documents/Tesis/Datasets/txts/alone")
# Goal de cualquier equipo as+bs
goal_as = mark_pre_goal(goal_as+goal_bs, seconds_in_future=2)
store_dataset_for_tf_as_cifar(goal_as, screens,"/home/eitan/Documents/Tesis/Datasets/bin", new_every_n_lines=16000)