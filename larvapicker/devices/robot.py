"""
This handles the "hand" of the LarvaPicker, aka the robot

Usage:
    Lydia.py


Server Connections:
    --inbound=HOST:PORT       Connection for inbound messages.
                              [default: localhost:5000]
    --outbound=HOST:PORT      Binding for outbound messages.
                              [default: *:5002]
"""

import cv2
import zmq
import time
import numpy as np
from pathlib import Path

from ..config.constants import *
from ..methods import robot as rm
from ..utils.publisher import Publisher
from ..utils.subscriber import Subscriber
from ..utils.poller import Poller
from ..utils.robot import Robot


class Lydia:
    def __init__(self):
        print('Loading resources...')

        # Initializing pub, sub, and poller ports
        context = zmq.Context()
        self.publisher = Publisher(name='rbt', port='5002', context=context)
        self.subscriber = Subscriber(name='rbt', port='5000', context=context)
        self.poller = Poller()
        self.poller.register(self.subscriber)
        print('ZMQ objects: '
              '\t\t\t\t\tDONE'
              '\t\t[...       ]')

        # Check that calibration files and image file exist
        if not h_file.exists() or not z_file.exists():
            print(f'******* ERROR: Could not find calibration files or image file.'
                  f'\n{h_file}\n{z_file}')
            exit()

        # Import calibration files into matrices
        self.H = np.load(h_file, allow_pickle=True)
        self.Z = np.load(z_file, allow_pickle=True)
        self.z_buffer = 0
        print('Calibration parameters: \t\tDONE\t\t[......    ]')

        # Initializing robot
        self.robot = Robot()

        rm.air_check(self.robot, True)
        rm.vac_check(self.robot, True)
        self.robot.vacuum_off()
        self.robot.air_off()
        print('Robot initialized: \t\t\tDONE\t\t[...........]')

        print('\n\n*** LYDIA IS READY FOR ACTION ***\n\n')

    def destroy(self):
        self.robot.home()
        self.robot.vacuum_off()
        self.robot.air_off()
        self.robot.close()
        exit()

    def read(self, cmd, args):
        if cmd == 'destroy':
            self.destroy()

        elif cmd == 'home_robot':
            self.home_robot()

        elif cmd == 'h_cal':
            self.h_cal()

        elif cmd == 'z_cal':
            self.z_cal()

        elif cmd == 'get_water':
            self.get_water(*np.array(args, dtype=int))

        elif cmd == 'pick_larva':
            self.pick_larva(int(args[0]), int(args[1]), float(args[2]))

        elif cmd == 'drop_larva':
            self.drop_larva(*np.array(args, dtype=int))

        elif cmd == 'h_recal':
            self.h_recal(*np.array(args, dtype=int))

        elif cmd == 'h_cal_reset':
            self.h_cal_reset()
            self.home_robot()

        elif cmd == 'deliver_to_larva':
            self.deliver_to_larva(*np.array(args, dtype=int))

        elif cmd == 'hold_larva':
            self.hold_larva()

        elif cmd == 'release_larva':
            self.release_larva()

        else:
            print('******* ERROR: UNRECOGNIZED REQUEST! Skipping.')

    def home_robot(self):
        self.robot.home()
        self.robot.air_on()
        time.sleep(0.2)
        self.robot.air_off()
        print('Ready for next image update.')

    def get_water(self, larva_x, larva_y):
        """
        This function maps the sequence of steps for getting water before picking up larvae for the robot.
        Given an initial larva position, this calculates the nearest water moat position based on the MoatFinder calibration,
        and actually moves the robot to that position.
        If necessary, it also moves the robot out of the way of the camera so it can properly refresh the larva position.

        :param larva_x, larva_y: position of the larva
        :return: None, but sends out request to refresh image
        """

        x, y = rm.get_xytransform(larva_x, larva_y, self.H)
        moat_x, moat_y = rm.get_nearest_moat_xy(x, y)
        self.robot.set_xy(moat_x, moat_y)
        self.robot.air_on()
        time.sleep(0.1)
        self.robot.air_off()
        self.robot.set_z(-22.5)
        time.sleep(0.1)
        self.robot.set_z(0)

        rm.move_out(self.robot, moat_x, moat_y)
        time.sleep(0.5)
        print('Arrived at moat.')
        time.sleep(1.0)
        self.publisher.send(f'lp crop_img {larva_x} {larva_y}')

    def pick_larva(self, larva_x, larva_y, larva_a):
        """
        This function maps the sequence of steps for picking up larvae for the robot.
        Given a refreshed larva position after dipping into water, this moves the robot to that position and picks up the larva.

        :param larva_x, larva_y: position of the larva
        :return: None, but sends out request to refresh image
        """

        x, y = rm.get_xytransform(larva_x, larva_y, self.H)

        # calculates surface height at the larva position
        z = rm.get_z_height(x, y, larva_a, self.Z)
        print(f'Larva coordinates: ({x:.3f}, {y:.3f}, {z:.3f})')

        # picks up larva
        self.robot.set_xy(x, y)
        self.robot.set_z(z)
        self.robot.vacuum_on()
        time.sleep(0.4)
        self.robot.set_z(z + 1.0)
        self.robot.vacuum_off()
        self.robot.set_z(-13)
        rm.move_out(self.robot, x, y)
        time.sleep(0.75)
        self.publisher.send(f'lp crop_img {larva_x} {larva_y}')

    def drop_larva(self, dest_x, dest_y):
        """
        This function maps the sequence of steps for dropping off larvae for the robot.
        It moves the larva to an empty spot in the middle of the agar as calculated by the brain (LarvaPicker.py).
        In case the initial drop-off via air pressure does not work, it also has a safeguard protocol that slides the
        nozzle along the agar to brush off any larvae that remains stuck on it.

        :param dest_x, dest_y: position of the drop off point
        :return: None, but sends out request to refresh entire image
        """

        x, y = rm.get_xytransform(dest_x, dest_y, self.H)

        # calculates surface height at the drop off position
        z = rm.get_z_height(x, y, None, self.Z)
        print(f'Larva coordinates: ({x:.3f}, {y:.3f}, {z:.3f})')

        self.robot.set_xy(x + 6, y)
        self.robot.set_z(z)
        self.robot.vacuum_off()

        self.robot.air_on()
        time.sleep(0.5)
        self.robot.slow_xy(x + 10, y, 400)
        self.robot.slow_xy(x, y + 10, 400)
        self.robot.slow_xy(x - 10, y, 400)
        self.robot.slow_xy(x, y - 10, 400)
        time.sleep(0.5)

        self.robot.set_z(-5)
        self.robot.air_off()
        self.robot.set_xy(x, 2)
        time.sleep(1.0)

        self.publisher.send(f'lp get_new_img {dest_x} {dest_y}')

    def deliver_to_larva(self, larva_x, larva_y, placebo=0):
        payload_x, payload_y, payload_z = solution_coordinates
        x, y = rm.get_xytransform(larva_x, larva_y, self.H)
        z = rm.get_z_height(x, y, None, self.Z)
        if placebo == 1:
            payload_x, payload_y = rm.get_nearest_moat_xy(x, y)
            z = -22.5

        # picks up solution
        self.robot.set_xy(payload_x, payload_y)
        self.robot.set_z(payload_z)
        time.sleep(0.1)
        self.robot.set_z(0)
        time.sleep(0.1)

        # spray the solution over the larva
        self.robot.set_xy(x, y)
        self.robot.set_z(z + 3.0)
        self.robot.air_on()
        time.sleep(0.5)
        self.robot.air_off()
        self.robot.set_z(0)

        # clean off nozzle and home the robot
        moat_x, moat_y = rm.get_nearest_moat_xy(x, y)
        self.robot.set_xy(moat_x, moat_y)
        self.robot.set_z(-22)
        self.robot.air_on()
        time.sleep(1.0)
        self.robot.set_z(0)
        self.robot.air_off()
        self.home_robot()

    def h_cal(self):
        """
        Based on the calibration design painted on the moat as captured by RobotCalibrator.py, it calculates the
        corresponding coordinates in robot units (from pixels) based on a previous calibration. The robot moves to each
        of these points and prompts the user to verify/modify each position to precisely recalibrate the homography
        transformation between camera and robot.
        Coordinates in both camera pixels and robot units are saved as separate .npy files for future use.
        The calculated homography matrix is also saved as .npy file and loaded.

        :return: None. New homography calibration .npy is saved and loaded, and a request for continuing calibration is sent out.
        """

        h_cam_list_temp = np.load(h_cam_file_temp, allow_pickle=True)
        h_rbt_list_temp = rm.get_homography(self.robot, h_cam_list_temp, self.H)

        h_cam_list = self.H[:, :2]
        h_rbt_list = self.H[:, 2:]
        a = input('Append calibration points to existing file? [Y/N] ')
        if a in ['Y', 'y', '']:
            h_cam_list_temp = np.append(
                h_cam_list_temp,
                np.array(h_cam_list)[:256, :],
                axis=0
            )
            h_rbt_list_temp = np.append(
                h_rbt_list_temp,
                np.array(h_rbt_list)[:256, :],
                axis=0
            )
        else:
            h_cam_list_temp = np.append(
                h_cam_list_temp,
                np.array(h_cam_list)[:32, :],
                axis=0
            )
            h_rbt_list_temp = np.append(
                h_rbt_list_temp,
                np.array(h_rbt_list)[:32, :],
                axis=0
            )

        h = np.append(h_cam_list_temp, h_rbt_list_temp, axis=-1)
        print(f'Total number of calibration points saved: \t{h.shape[0]}')

        t = input(f'Save new calibration points to file: {h_file}? [Y/N] ')
        if t in ['y', 'Y', '']:
            np.save(h_file, h)
            print('Saved homography. Loading new homography file.')
            self.H = h
            self.publisher.send('lp init_z_cal')
        else:
            print('Save canceled, homography deleted.')
            self.publisher.send('lp init_h_cal')

    def h_recal(self, cam_x, cam_y):
        """
        Upon manual trigger, this function allows for a new calibration point to be added to the homography calculation.
        The new homography is saved as .npy file and loaded.

        :param cam_x, cam_y: position (in pixels) to recalibrate at
        :return: None. New homography calibration is saved and loaded. Request to restart LarvaPicker process is sent.
        """

        h_cam_list = self.H[:, :2]
        h_rbt_list = self.H[:, 2:]
        rdx = np.argsort(np.linalg.norm(h_cam_list - np.array([[cam_x, cam_y]]), axis=-1))
        rand_idx = rdx[:16]
        np.random.shuffle(rand_idx)
        rdx = np.append(rdx[16:], rand_idx[:8])

        h = np.append(h_cam_list[rdx, :], h_rbt_list[rdx, :], axis=-1)
        print(f'Updating calibration points...'
              f'\nRemaining total number of points: {h.shape[0]}')
        self.H = h

    def h_cal_reset(self):
        print(f'Reset homography to original calibration on file:\n{h_file}')
        self.H = np.load(h_file, allow_pickle=True)

    def z_cal(self):
        """
        The robot moves to each point in a predefined grid of calibration points (currently a 7x7 grid) and finds the
        Z height of the agar by slowly lowering the nozzle until it detects a pressure change due to contact with the
        agar surface.
        The coordinates of the calibration points and the corresponding height at each point are saved as .npy file.

        :return: None. New height calibration .npy is saved and loaded, a request for continuing calibration is sent out.
        """
        # Lets the air flush out the nozzle first
        # Then checks if the vacuum pressure is high enough to perform the calibration
        p_threshold = 49.0
        p_init = True
        while p_init:
            print('Checking vacuum pressure')
            self.robot.vacuum_off()
            self.robot.air_on()
            time.sleep(1.0)
            p_check = rm.vac_check(self.robot, False)
            print(f'Pressure at: {p_check}')
            if 35 < p_check < p_threshold:
                p_init = False
                print(f'Vacuum pressure check: OK'
                      f'\nVac reading: {p_check}.'
                      f'\nSetting threshold to: {p_threshold}')
            else:
                print('Please adjust vacuum pressure.')
                time.sleep(2.0)

        measured_heights = rm.get_height_grid(self.robot, p_threshold)
        print(f'Measured heights:\n{measured_heights}')
        t = input(f'Save new z_map to file: {z_file} ? [Y/n] ')
        if t in ['y', 'Y', '']:
            np.save(z_file, measured_heights)
            print('Saved. Loading new height map file.')
            self.Z = measured_heights
            self.publisher.send('lp finish')
        else:
            print('Save canceled, z_map deleted.')
            self.publisher.send('lp init_z_cal')

    def hold_larva(self):
        x, y, z = dormitory_coordinates
        self.robot.set_xy(x, y)
        self.robot.set_z(z)
        self.robot.vacuum_off()
        self.robot.air_on()
        time.sleep(0.2)
        self.robot.set_z(z + 1.0)
        self.robot.air_off()
        time.sleep(0.5)
        self.robot.set_z(-5)
        self.robot.set_xy(x, 4)
        time.sleep(1.0)

    def release_larva(self):
        x, y, z = dormitory_coordinates
        self.robot.set_xy(16, y + 30)
        self.robot.set_z(-22.5)
        time.sleep(0.3)
        self.robot.set_z(-10)

        self.robot.set_xy(x, y)
        time.sleep(0.1)
        self.robot.set_z(z)
        self.robot.vacuum_on()
        time.sleep(0.3)
        self.robot.set_z(z + 1.0)
        self.robot.vacuum_off()
        self.robot.set_z(-10)
        time.sleep(0.5)

        self.drop_larva(1000, 1000)

    def run(self):
        while True:
            if self.poller.poll(self.subscriber):
                cmd, args = self.subscriber.recv()
                self.read(cmd, args)


def main():
    rbt = Lydia()
    rbt.run()


if __name__ == '__main__':
    main()
