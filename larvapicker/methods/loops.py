import numpy as np
import random
import time
import cv2

from ..config.constants import *
from .images import *


def standby(image, img_str, log_idx, publisher, current_time, instar):
    img_c = clean_img(image, log_idx)
    larva_list, cntr_list = get_contours(img_c, instar)
    n = len(larva_list)
    print(f'Found {n} larvae.')
    if n == 0:
        line = f'0 0 NONE'
    elif cntr_list is not None and n == 1:
        line = f'{larva_list[0][0]} {larva_list[0][1]} 1'
    else:
        line = f'{n} {n} {n}'
    publisher.send(f'log write_to_memory {line} {current_time} {img_str}', prt=False)


def start_loop(x, y, publisher, current_time):
    publisher.send(f'cam set_ready_check 0')
    publisher.send(f'rbt get_water {x} {y}')
    line = f'{x} {y} INIT'
    publisher.send(f'log write_to_memory {line} {current_time}')


def pick_up(x, y, a, publisher, current_time):
    publisher.send(f'cam set_larva_xy {x} {y}')
    publisher.send(f'rbt pick_larva {x} {y} {a}')
    line = f'{x} {y} PICKUP'
    publisher.send(f'log write_to_memory {line} {current_time}')


def drop_off(n, center_idx, publisher, current_time):
    img_c = cv2.imread(img_file, 0)  # Load image from file as greyscale
    while img_c is None:
        img_c = cv2.imread(img_file, 0)  # Blocks until the image is properly loaded

    img_c_c = clean_img(img_c, center_idx)
    destination_list = get_available_space(img_c_c)

    if len(destination_list) < n + 1:
        print('Error: not enough destination squares.')
        for j in range(n - len(destination_list) + 2):
            destination_list.append(np.array([int(img_c.shape[1] / 2), int(img_c.shape[0] / 2)], float))

    i = random.randint(0, len(destination_list) - 1)
    publisher.send(f'rbt drop_larva {destination_list[i][0]} {destination_list[i][1]}')
    line = f'{destination_list[i][0]} {destination_list[i][1]} DROPOFF'
    publisher.send(f'log write_to_memory {line} {current_time}')


def end_loop(publisher):
    publisher.send('rbt home_robot')
    time.sleep(5.0)
    publisher.send('cam set_ready_check 1')
