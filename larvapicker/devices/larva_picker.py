"""
This handles the "brain" of the LarvaPicker, aka the data processor

Usage:
    LarvaPicker.py


Server connections:
    --inbound=HOST:PORT       Connect for inbound messages from camera and robot.
                              [default: localhost:5001 and localhost:5002]
    --outbound=HOST:PORT      Binding for outbound messages.
                              [default: *:5000]
"""

import numpy as np
import cv2
import zmq
import time
from pathlib import Path

from ..config.constants import *
from ..methods import images as im
from ..methods import loops as lm
from ..utils.publisher import Publisher
from ..utils.subscriber import Subscriber
from ..utils.poller import Poller


class LarvaPicker:
    def __init__(self):
        # Manually input instar of the larvae
        self.instar = int(input('Select instar: 1, 2, or 3? '))
        while self.instar not in [1, 2, 3, 4]:
            self.instar = int(input('******* ERROR!'
                                    '\nPlease select proper instar: 1, 2, or 3? '))

        print('Loading resources...')

        context = zmq.Context()
        self.publisher = Publisher(name='lp', port='5000', context=context)
        self.subscriber_cam = Subscriber(name='lp', port='5001', context=context)
        self.subscriber_rbt = Subscriber(name='lp', port='5002', context=context)
        self.poller = Poller()
        self.poller.register(self.subscriber_cam)
        self.poller.register(self.subscriber_rbt)
        print('ZMQ objects: \t\t\tDONE\t\t[...       ]')

        mask_p = np.zeros((cam_h, cam_w), int)
        mask_p[stage_size:cam_h - stage_size,
               stage_size:cam_w - stage_size] = 1
        mask_p[stage_size + margin_size:cam_h - stage_size - margin_size,
               stage_size + margin_size:cam_w - stage_size - margin_size] = 0
        self.mask_idx_p = (mask_p == 0)

        mask_c = np.zeros((cam_h, cam_w), int)
        mask_c[(cam_h - center_size)//2:(cam_h + center_size)//2,
               (cam_w - center_size)//2:(cam_w + center_size)//2] = 1
        self.mask_idx_c = (mask_c == 0)

        mask_l = np.zeros((cam_h, cam_w), int)
        mask_l[stage_size:cam_h - stage_size,
               stage_size:cam_w - stage_size] = 1
        self.mask_idx_l = (mask_l == 0)

        mask_f = np.zeros((cam_h, cam_w), int)
        mask_f[stage_size + margin_size:cam_h - stage_size - margin_size,
               stage_size + margin_size:cam_w - stage_size - margin_size] = 1
        self.mask_idx_f = (mask_f == 0)
        print('Image masks: \t\t\tDONE\t\t[........  ]')

        self.ready = True
        self.hold = False
        self.prev_count = -1
        self.miss_count = 0

        self.pick_up_count = [0, 0, 0]
        self.drop_off_count = [0, 0]
        self.pick_up_fail_count = 0
        self.drop_off_fail_count = 0
        print('States and constants: \tDONE\t\t[..........]')

        time.sleep(1.0)
        self.publisher.send('cam set_ready_check 1')
        self.start_time = time.time()
        print(f'Start time set at: {self.start_time}'
              f'\n*** STARTING LarvaPicker PROCESS ***\n\n')

    def destroy(self):
        """
        Ends all processes by sending exit commands to the other processes and exiting out of this one
        :return: None.
        """
        self.publisher.send(f'cam destroy')
        self.publisher.send(f'rbt destroy')
        self.publisher.send(f'log destroy '
                            f'{self.pick_up_count[0]} {self.pick_up_count[1]} {self.pick_up_count[2]} '
                            f'{self.drop_off_count[0]} {self.drop_off_count[1]} '
                            f'{self.pick_up_fail_count} {self.drop_off_fail_count}')
        print('Exiting all processes.')
        exit()

    def read(self, cmd, args):
        if cmd == 'destroy':
            self.destroy()

        elif cmd == 'search_img':
            if args is None:
                self.ready = True
                self.publisher.send('cam set_ready_check 1')
                self.publisher.send('rbt h_cal_reset')
            elif self.ready:
                self.search_img(' '.join(args))
            else:
                print('******* MISMATCHED COMMAND RECEIVED. Trying to resync.')
                self.publisher.send('cam set_ready_check 0')

        elif cmd == 'search_crop_img':
            self.search_crop_img(*np.array(args[:2], dtype=int), ' '.join(args[2:]))

        elif cmd == 'search_new_img':
            self.search_new_img(*np.array(args[:2], dtype=int), ' '.join(args[2:]))

        elif cmd == 'crop_img':
            self.publisher.send(f'cam crop_img {int(args[0])} {int(args[1])}')

        elif cmd == 'get_new_img':
            time.sleep(0.7)
            self.publisher.send(f'cam new_img {int(args[0])} {int(args[1])}')

        elif cmd == 'h_recal':
            self.publisher.send(f'rbt h_recal {int(args[0])} {int(args[1])}')
            time.sleep(3.0)
            self.ready = True
            self.publisher.send('cam set_ready_check 1')

        elif cmd == 'deliver_to_larva':
            self.deliver_to_larva(' '.join(args))
            time.sleep(35.0)
            self.publisher.send('cam set_ready_check 1')
            self.ready = True

        elif cmd == 'hold_larva':
            self.hold_larva(' '.join(args))

        elif cmd == 'release_larva':
            self.publisher.send('rbt release_larva')
            # time.sleep(35.0)
            # self.publisher.send('eye set_ready_check 1')
            self.hold = False
            self.ready = False

        elif cmd == 'pause_process':
            if self.ready:
                self.ready = False
            else:
                self.ready = True

        else:
            print('******* ERROR: UNRECOGNIZED REQUEST! Skipping.')

    def search_img(self, img_str):
        """
        Reads latest image as written by Basler and parses it to look for any larvae at the perimeter.
        If larva detected at perimeter, halts standby recording loop and sends request for Lydia to begin pick up process.

        :return: None. Sends request to record current activity and larva state.
        """
        img = im.str2img(img_str)
        current_time = time.time() - self.start_time
        print(f'Updated @ {current_time}')

        # if current_time % 3600 < 1.0 and current_time >= 3600:
        #     self.publisher.send('eye set_ready_check 0')
        #     if current_time % 10800 < 1.0:
        #         self.deliver_to_larva(img_str, True)
        #     else:
        #         self.deliver_to_larva(img_str)
        #     time.sleep(20.0)
        #     self.publisher.send('eye set_ready_check 1')
        #     self.ready = True
        #     return

        img_p = im.clean_img(img, self.mask_idx_p)
        larva_list, cntr_list = im.get_contours(img_p, self.instar)
        n = len(larva_list)
        if n > 0:
            print(f'******* DETECTED {n} LARVAE AT PERIMETER.'
                  f'\n******* INITIATING ROBOT.')
            lm.start_loop(
                larva_list[0][0], larva_list[0][1],
                self.publisher,
                current_time
            )
            self.ready = False
        else:
            lm.standby(
                img, img_str,
                self.mask_idx_l,
                self.publisher,
                current_time,
                self.instar
            )

    def search_crop_img(self, larva_x, larva_y, crop_str):
        """
        Reads latest cropped image as written by Basler and parses it to look for larvae in the image.
        Depending on when and how this function was requested, it can do the following:
        1. Larva has been picked up if there are less larva in the cropped image than before. Sends request for drop off.
        2. None detected. Reinitiates standby recording loop.
        3. Larva pick up attempt was unsuccessful if there are as many larvae as before. Sends request for pick up.
        4. If no previous activity is recorded and a larva is detected in image, sends request for pick up.

        :param larva_x, larva_y: position of larva in the cropped image
        :return: None. Various requests can be sent.
        """

        crop_img = im.str2img(crop_str)
        current_time = time.time() - self.start_time
        print(f'Refreshed @ {current_time}')
        crop_img_p = im.clean_img(crop_img, None)
        larva_list, cntr_list = im.get_contours(crop_img_p, self.instar)
        n = len(larva_list)
        print(f'Found {n} larvae in cropped vicinity.')

        if n < self.prev_count:
            print('Larva picked up. Dropping off larva.')
            self.pick_up_count[self.miss_count] += 1
            self.miss_count = 0
            if self.hold is True:
                line = f'{0} {0} FEED'
                self.publisher.send(f'log write_to_memory {line} {current_time}')
                self.publisher.send(f'rbt hold_larva')
            else:
                lm.drop_off(
                    n,
                    self.mask_idx_c,
                    self.publisher,
                    current_time
                )
            n = -1

        elif n == 0:
            print('No larva found in cropped vicinity. Skipping.')
            lm.end_loop(self.publisher)
            self.ready = True
            self.miss_count = 0

        elif n == self.prev_count:
            self.miss_count += 1
            if self.miss_count >= 3:
                print('Pick up attempt unsuccessful.'
                      '\n******* MAX ATTEMPTS REACHED. Interrupting loop.')
                self.publisher.send(f'cam add_ignore '
                                    f'{larva_list[0][0] + larva_x - crop_size} '
                                    f'{larva_list[0][1] + larva_y - crop_size}')
                self.publisher.send(f'rbt h_recal '
                                    f'{larva_list[0][0] + larva_x - crop_size} '
                                    f'{larva_list[0][1] + larva_y - crop_size}')
                time.sleep(0.5)
                lm.end_loop(self.publisher)
                self.ready = True
                self.pick_up_fail_count += 1
                self.miss_count = 0
            else:
                print('Pick up attempt unsuccessful. Trying again.')
                time.sleep(0.5)
                lm.start_loop(
                    larva_list[0][0] + larva_x - crop_size,
                    larva_list[0][1] + larva_y - crop_size,
                    self.publisher,
                    current_time
                )
            n = -1

        else:
            print(f'Refreshed image for {n} larvae. Picking larva.')
            lm.pick_up(
                larva_list[0][0] + larva_x - crop_size,
                larva_list[0][1] + larva_y - crop_size,
                cv2.contourArea(cntr_list[0]),
                self.publisher,
                current_time
            )

        self.prev_count = n

    def search_new_img(self, dest_x, dest_y, img_str):
        """
        After a successful pick up and drop off, the newest image from Basler is read and parsed before reinitiating
        standby loop.
        This is to save time home-ing the robot between pick ups in case of multiple larvae at the perimeter.

        :return: None. Sends request for new pick up or reinitiates standby loop.
        """

        img_n = im.str2img(img_str)
        current_time = time.time() - self.start_time
        crop = im.crop_img(img_n, dest_x, dest_y, crop_size * 4)
        crop_c = im.clean_img(crop, None)
        larva_list, cntr_list = im.get_contours(crop_c, self.instar)
        n = len(larva_list)
        print(f'Found {n} larvae in cropped vicinity.')

        if n >= 1:
            print('Drop off successful. Checking perimeter.')
            self.drop_off_count[self.miss_count] += 1
            self.miss_count = 0
            # Parse image to find larva at the perimeter
            img_p = im.clean_img(img_n, self.mask_idx_p)
            larva_list, cntr_list = im.get_contours(img_p, 1)

            n = len(larva_list)
            if n == 0:
                print('All larvae within perimeter. Ready for next image update.')
                lm.end_loop(self.publisher)
                self.ready = True
            else:
                print(f'******* DETECTED {n} LARVAE AT PERIMETER.'
                      f'\n******* INITIATING ROBOT.')
                lm.start_loop(
                    larva_list[0][0], larva_list[0][1],
                    self.publisher,
                    current_time
                )

        else:
            self.miss_count += 1
            if self.miss_count >= 2:
                print('Drop off unsuccessful.'
                      '\n******* MAX ATTEMPTS REACHED. Interrupting loop.')
                lm.end_loop(self.publisher)
                self.ready = True
                self.drop_off_fail_count += 1
                self.miss_count = 0
            else:
                print('Drop off unsuccessful. Trying again.')
                lm.drop_off(
                    n,
                    self.mask_idx_c,
                    self.publisher,
                    current_time
                )

    def deliver_to_larva(self, img_str, placebo=False):
        img = im.str2img(img_str)
        img_f = im.clean_img(img, self.mask_idx_f)
        larva_list, cntr_list = im.get_contours(img_f, self.instar)
        n = len(larva_list)
        print(f'Found {n} larvae.')
        if n > 0:
            current_time = time.time() - self.start_time
            line = f'{0} {0} DELIVER'
            self.publisher.send(f'log write_to_memory {line} {current_time}')

            if placebo:
                self.publisher.send(f'rbt deliver_to_larva {larva_list[0][0]} {larva_list[0][1]} 1')
            else:
                self.publisher.send(f'rbt deliver_to_larva {larva_list[0][0]} {larva_list[0][1]}')
            self.ready = False

        else:
            lm.end_loop(self.publisher)
            self.ready = True

    def hold_larva(self, img_str):
        img = im.str2img(img_str)
        img_f = im.clean_img(img, self.mask_idx_f)
        larva_list, cntr_list = im.get_contours(img_f, self.instar)
        n = len(larva_list)
        print(f'Found {n} larvae.')
        if n > 0:
            current_time = time.time() - self.start_time
            lm.start_loop(
                larva_list[0][0], larva_list[0][1],
                self.publisher,
                current_time
            )
            self.ready = False
            self.hold = True
        else:
            lm.end_loop(self.publisher)
            self.ready = True
            self.hold = False

    def run(self):

        print('Beginning larva picker process. Please make sure robot and camera are both running.')

        while True:
            if self.poller.poll(self.subscriber_cam):
                cmd, args = self.subscriber_cam.recv()
                self.read(cmd, args)

            if self.poller.poll(self.subscriber_rbt):
                cmd, args = self.subscriber_rbt.recv()
                self.read(cmd, args)


def main():
    lp = LarvaPicker()
    lp.run()


if __name__ == '__main__':
    main()
