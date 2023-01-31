import numpy as np
import scipy.spatial as sp
import cv2
import time
from scipy.interpolate import griddata

from ..config.constants import *


def air_check(robot, manual=False):
    robot.vacuum_off()
    robot.air_on()
    time.sleep(1.0)
    while manual:
        p = robot.get_pressure()
        g = input(f'Current pressure reading: {p}. Move onto vacuum pressure? ')
        if g in ['Y', 'y', '']:
            print(f'Air pressure set at: {p}')
            break

    p = robot.get_pressure()

    return p


def vac_check(robot, manual=False):
    robot.air_off()
    robot.vacuum_on()
    time.sleep(1.0)
    while manual:
        p = robot.get_pressure()
        g = input(f'Current pressure reading: {p}. Pressures all set? ')
        if g in ['Y', 'y', '']:
            print(f'Vacuum pressure set at: {p}')
            break

    p = robot.get_pressure()

    return p


def move_out(robot, x, y):
    print('Moving out of frame.')
    if y < 100:
        y_set = 5
    elif 100 <= y < 170:
        y_set = y - 90
    elif y >= 170:
        if x > 210:
            y_set = y - 120
        else:
            y_set = 234
    else:
        y_set = y
    robot.set_xy(x, y_set)


def get_homography(robot, cam_coord, H):
    interval = 0.1
    robot_coord = []
    cal_image = cv2.imread(h_img_file)
    cv2.namedWindow('Robot Controller')
    print('Click on \'Robot Controller\' window and use WASD to move the robot',
          'until it is directly above the calibration point.',
          '\nPress q or esc to exit and move onto next calibration point.')

    for [x_cam, y_cam] in cam_coord:
        x_robot, y_robot = get_xytransform(x_cam, y_cam, H)
        local_image = cal_image.copy()
        cv2.circle(local_image, (int(x_cam), int(y_cam)),
                   radius=5, color=colour_list[2], thickness=-1)
        cv2.circle(local_image, (int(x_cam), int(y_cam)),
                   radius=23, color=colour_list[2], thickness=4)
        scaled = cv2.resize(local_image, (cam_w // 2, cam_h // 2))
        cv2.imshow('Robot Controller', scaled)

        robot.set_xy(x_robot, y_robot)
        robot.set_z(-19)
        key = -1
        while key != ord('q'):
            x, y, z = robot.get_position()
            key = cv2.waitKey(10) & 0xFF
            if key == ord('w'):
                robot.set_xy(x + interval, y)
            elif key == ord('a'):
                robot.set_xy(x, y - interval)
            elif key == ord('s'):
                robot.set_xy(x - interval, y)
            elif key == ord('d'):
                robot.set_xy(x, y + interval)

        robot.set_z(0)
        x_final, y_final, z_final = robot.get_position()
        robot_coord.append([x_final, y_final])

    robot.home()
    robot_coord = np.asarray(robot_coord)
    cv2.destroyWindow('Robot Controller')

    return robot_coord


def get_xytransform(x_cam, y_cam, H):
    x_rbt, y_rbt = griddata(H[:, :2], H[:, 2:], [[x_cam, y_cam]], method='linear')[0]
    if np.isnan(x_rbt) or np.isnan(y_rbt):
        rdx = np.argsort(np.linalg.norm(H[:, :2] - np.array([[x_cam, y_cam]]), axis=-1))
        h, status = cv2.findHomography(H[rdx[:16], :2], H[rdx[:16], 2:], cv2.LMEDS)
        xy_list = cv2.perspectiveTransform(np.array([[[x_cam, y_cam]]], dtype=np.float32), h)
        x_rbt, y_rbt = xy_list[0][0]

    return x_rbt, y_rbt


def get_nearest_moat_xy(x, y):
    """
    Given coordinates (x, y) in robot units and the moat calibration vector, calculates the coordinates of the nearest
    moat position.

    :param x, y: coordinates to analyze
    :return: coordinates of nearest moat position
    """

    if x < y <= max_robot_y - x or (y == max_robot_y / 2 and x == max_robot_x / 2):
        # Bottom edge
        x_moat = 16.0
        if y < 40.0:
            y_moat = 40.0
        else:
            y_moat = y
        print('Moving to bottom edge')
    elif max_robot_y - x < y <= x:
        # Top edge
        x_moat = 245.0
        y_moat = y
        print('Moving to top edge')
    elif y >= x and y > max_robot_y - x:
        # Right edge
        x_moat = x
        y_moat = 234.5
        print('Moving to right edge')
    elif y < x and y <= max_robot_y - x:
        # Left edge
        x_moat = x
        y_moat = 8.0
        print('Moving to left edge')
    else:
        # By default, moves to left edge
        x_moat = x
        y_moat = 8.0
        print('Couldn\'t find nearest edge. By default, moving to left edge.')

    return x_moat, y_moat


def get_height_grid(robot, p_threshold):
    # The calibration grid
    y_grid, x_grid = np.meshgrid(
        np.array([30, 45, 80, 120, 160, 195, 210]),
        np.array([40, 60, 100, 130, 160, 200, 220])
    )

    # Configuration parameters
    z_initial = -14.0
    z_increment = 0.1

    # At each calibration point, the nozzle moves down slowly while checking the pressure
    # Upon detecting a pressure spike due to contact with surface, the program logs the current height
    # and moves onto the next calibration point
    z_measured_heights = np.empty((0, 3))
    for x, y in zip(x_grid.ravel(), y_grid.ravel()):
        print(f'Point: ({x} {y})')
        z_current = z_initial
        seek = True

        # homes robot after each row
        if y == y_grid[0, 0]:
            robot.home()
            robot.air_on()
            time.sleep(1.5)
            robot.air_off()
            robot.vacuum_on()
            time.sleep(0.5)
            z_initial = z_current + 4.0

        time.sleep(0.2)
        robot.set_xy(x, y)
        time.sleep(0.2)
        robot.set_z(z_initial)
        robot.vacuum_on()
        time.sleep(0.5)

        p = robot.get_pressure()
        while p > p_threshold - 1.0:
            print(f'******* Initial vac reading too high: {p}')
            robot.set_z(z_initial + 2.5)
            robot.vacuum_off()
            robot.air_on()
            time.sleep(1.5)
            robot.air_off()
            robot.vacuum_on()
            time.sleep(0.5)
            p = robot.get_pressure()
            print(f'******* Vac reading reset to: {p}')

        while seek is True:
            robot.set_z(z_current)

            while True:
                time.sleep(0.25)
                p = robot.get_pressure()
                dp = p - robot.get_pressure()
                if dp < 0.2:
                    break

            if p > p_threshold:
                print(f'Vac reading at: {p} '
                      f'-- Found surface at ({x}, {y}, {z_current:.2f}).')
                z_measured_heights = np.append(z_measured_heights, [[x, y, z_current]], axis=0)
                z_initial = z_current + 2.5
                robot.vacuum_off()
                robot.set_z(0.0)
                seek = False
            else:
                z_current = z_current - z_increment
                if z_current < -23:
                    print(f'*** No surface found at ({x}, {y})')
                    z_measured_heights = np.append(z_measured_heights, [[x, y, z_current]], axis=0)
                    seek = False

    robot.vacuum_off()
    robot.home()

    return z_measured_heights


z_buffer = 0


def get_z_height(x, y, area, Z):
    """
    Given a height map from ZHeightCal, this function calculates Z height of the agar bed at any X/Y coordinate.
    This interpolates the height at the given point from the grid of heights measured from calibration.

    :param ptX, ptY: position at which to calculate the height
    :return: the interpolated height at (ptX, ptY)
    """
    global z_buffer

    z = griddata(Z[:49, :2], Z[:49, 2], (x, y), method='nearest')

    if area is not None:
        z_buffer = area / 600 + 0.4

    return z + z_buffer
