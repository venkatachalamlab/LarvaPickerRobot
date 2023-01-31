"""
This handles the "brain" of the LarvaPicker, aka the data processor
This is specifically to drive the camera and robot to fully calibrate in preparation for an experiment

Usage:
    RobotCalibrator.py


Server Connections:
    --inbound=HOST:PORT       Connect for inbound messages from camera and robot.
                              [default: localhost:5001 and localhost:5002]
    --outbound=HOST:PORT      Binding for outbound messages.
                              [default: *:5000]
"""

import numpy as np
import cv2
import zmq

from ..config.constants import *
from ..methods import images as im
from ..utils.publisher import Publisher
from ..utils.subscriber import Subscriber
from ..utils.poller import Poller


class RobotCalibrator:
    def __init__(self):
        context = zmq.Context()
        self.publisher = Publisher(name='lp', port='5000', context=context)
        self.subscriber_cam = Subscriber(name='lp', port='5001', context=context)
        self.subscriber_rbt = Subscriber(name='lp', port='5002', context=context)
        self.poller = Poller()
        self.poller.register(self.subscriber_cam)
        self.poller.register(self.subscriber_rbt)

    def destroy(self):
        print('Finished calibration. Exiting process.')
        exit()

    def read(self, cmd, args):
        if cmd == 'destroy' or cmd == 'finish':
            self.destroy()

        elif cmd == 'init_h_cal':
            self.init_h_cal()

        elif cmd == 'h_cal':
            self.h_cal(' '.join(args))

        elif cmd == 'init_z_cal':
            loop = input('Check new hography? [Y/N] ')
            if loop in ['Y', 'y', '']:
                self.publisher.send('cam h_img')
            else:
                self.init_z_cal()

        else:
            print('******* ERROR: UNRECOGNIZED REQUEST! Skipping.')

    def init_h_cal(self):
        init_h = input('Update homography? [Y/N] ')
        if init_h in ['Y', 'y', '']:
            self.publisher.send('cam h_img')
        else:
            self.init_z_cal()

    def h_cal(self, h_str):
        # Cleans up image to b/w and thresholded at a set value
        # Find corners using 'goodFeatureToTrack' from OpenCV
        # *** Adjust quality level and  threshold level to optimize corners detected *** #

        h_img = im.str2img(h_str, as_bw=True)
        h, w = h_img.shape
        crop = h_img[stage_size:h - stage_size, stage_size:w - stage_size]
        feature_list = cv2.goodFeaturesToTrack(
            crop,
            maxCorners=25,
            qualityLevel=0.20,
            minDistance=300
        )

        if feature_list is None:
            print('No corners found.')
            cam_corner_list = None
        else:
            # Reformatting the output corners vector
            cam_corner_list = feature_list[:, 0, :] + stage_size

            # Mark the corners on the original image
            h_img = cv2.cvtColor(h_img, cv2.COLOR_GRAY2BGR)
            for corner in cam_corner_list:
                cv2.circle(
                        h_img, tuple(corner.astype(int)),
                        radius=7, color=colour_list[2], thickness=-1
                )

            print(f'{len(cam_corner_list)} corners detected:\n')

        # Make a scaled copy to display
        # Full resolution image is written
        scaled = cv2.resize(h_img, (w//4, h//4))
        cv2.namedWindow('Calibration')
        cv2.imshow('Calibration', scaled)
        cv2.imwrite(h_img_file, h_img)

        # You can hit esc to continue in the image window to continue the calibration
        # Or hit 'e' if the corners detected look bad and you want to throw the image away and start over
        print('Click on image and hit esc to continue with calibration, or e to try again.')
        k = cv2.waitKey()
        if k == 27 and cam_corner_list is not None:
            print('Good image, continuing with calibration.')
            np.save(h_cam_file_temp, cam_corner_list)
            self.publisher.send('rbt h_cal')
            print('Check Lydia for calibration progress.')

        elif k == ord('e') or cam_corner_list is None:
            print('Bad image, trying new image.')
            self.publisher.send('cam h_img')

        cv2.destroyWindow('Calibration')

    def init_z_cal(self):
        init_z = input('Update height map? [Y/N] ')
        if init_z in ['y', 'Y', '']:
            self.publisher.send('rbt z_cal')
            print('Check Lydia for calibration progress.')
        else:
            self.destroy()

    def run(self):
        print('Beginning calibration process. Please make sure robot and camera are both running.')

        self.init_h_cal()

        while True:
            if self.poller.poll(self.subscriber_cam):
                cmd, args = self.subscriber_cam.recv()
                self.read(cmd, args)

            if self.poller.poll(self.subscriber_rbt):
                cmd, args = self.subscriber_rbt.recv()
                self.read(cmd, args)


def main():
    lp = RobotCalibrator()
    lp.run()


if __name__ == '__main__':
    main()
