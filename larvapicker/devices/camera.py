"""
This handles the camera for the LarvaPicker

Usage:
    Grasshopper.py


Server Connections:
    --inbound=HOST:PORT       Connection for inbound messages.
                              [default: localhost:5000]
    --outbound=HOST:PORT      Binding for outbound messages.
                              [default: *:5001]
"""

import numpy as np
import cv2
import zmq
import time
import PySpin

from ..config.constants import *
from ..methods import images as im
from ..utils.publisher import Publisher
from ..utils.subscriber import Subscriber
from ..utils.poller import Poller


class Grasshopper:
    def __init__(self):
        context = zmq.Context()
        self.publisher = Publisher(name='cam', port='5001', context=context)
        self.subscriber = Subscriber(name='cam', port='5000', context=context)
        self.poller = Poller()
        self.poller.register(self.subscriber)

        print(f'Loading resources...')

        for t in range(10):
            self.system = PySpin.System.GetInstance()
            self.cam_list = self.system.GetCameras()
            if self.cam_list.GetSize() > 0:
                print(f'Detected {self.cam_list.GetSize()} cameras.')
                break
            else:
                time.sleep(1)

        if self.cam_list.GetSize() == 0:
            print('******* ERROR: No camera detected! Exiting.')
            exit()

        self.cam = self.cam_list.GetByIndex(0)
        self.cam.Init()
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        sNodemap = self.cam.GetTLStreamNodeMap()
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        node_bufferhandling_mode.SetIntValue(PySpin.StreamBufferHandlingMode_NewestOnly)
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self.cam.ExposureTime.SetValue(4000)
        self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        self.cam.Gain.SetValue(0)
        self.cam.BlackLevel.SetValue(0)

        self.cam.BeginAcquisition()
        img = self.cam.GetNextImage()
        self.cam_h = img.GetHeight()
        self.cam_w = img.GetWidth()
        img_cv = cv2.cvtColor(img.GetNDArray(), cv2.COLOR_BAYER_BG2BGR)
        cv2.imwrite(img_file, img_cv)
        img.Release()

        print(f'Camera opened with image shape: ({img_cv.shape})'
              f'\nLoading config parameters.'
              f'\nImage file location: {img_file}')

        self.larva_x = 500
        self.larva_y = 500
        self.ready_check = False
        self.mask = np.ones((self.cam_h, self.cam_w, 3), int)
        self.mask[:stage_size, :] = [0, 0, 0]
        self.mask[self.cam_h - stage_size:, :] = [0, 0, 0]
        self.mask[:, :stage_size] = [0, 0, 0]
        self.mask[:, self.cam_w - stage_size:] = [0, 0, 0]
        self.mask_idx = (self.mask == 0)
        self.mask_display = np.zeros((self.cam_h//4, self.cam_w//4), int)

        print('Config parameters and resources loaded.'
              '\nPress SPACEBAR in image window to manually update image file.'
              '\nPress ESC in image window to end all processes.'
              '\n\n*** GRASSHOPPER IS NOW ACTIVE ***\n\n')
        self.times = [time.time()]

    def destroy(self):
        cv2.destroyAllWindows()
        self.cam.EndAcquisition()
        self.cam.DeInit()
        self.cam_list.Clear()
        del self.cam
        del self.cam_list
        self.system.ReleaseInstance()
        exit()

    def read(self, cmd, args):
        if cmd == 'destroy':
            self.destroy()

        elif cmd == 'read_img':
            self.process_next_frame(rc_override=True)

        elif cmd == 'write_img':
            self.write_img(img_file, save=True)

        elif cmd == 'h_img':
            h_img, h_str = self.write_img(h_img_file, ignore=False, as_bw=False)
            self.publisher.send(f'lp h_cal {h_str}')

        elif cmd == 'set_larva_xy':
            self.set_larva_xy(*np.array(args, dtype=int))

        elif cmd == 'set_ready_check':
            if args[0] == '1':
                self.ready_check = True
            else:
                img = self.read_img()
                cv2.imwrite(img_file, img)
                self.ready_check = False

        elif cmd == 'crop_img':
            self.set_larva_xy(*np.array(args, dtype=int))
            time.sleep(0.5)
            img = self.crop_img()
            cv2.imwrite(img_file, img)

        elif cmd == 'new_img':
            self.set_larva_xy(*np.array(args, dtype=int))
            n_img, n_str = self.write_img(img_file, save=True, ignore=True, as_bw=True)
            self.publisher.send(f'lp search_new_img {self.larva_x} {self.larva_y} {n_str}')

        elif cmd == 'add_ignore':
            x, y = np.array(args, dtype=int)
            self.mask[y-16:y+16, x-16:x+16] = [0, 0, 0]
            self.mask_idx = (self.mask == 0)
            self.mask_display[y//4-4:y//4+4, x//4-4:x//4+4] = 1

        else:
            print('******* ERROR: UNRECOGNIZED REQUEST! Skipping.')

    def read_img(self):
        img = self.cam.GetNextImage()
        img_cv = cv2.cvtColor(img.GetNDArray(), cv2.COLOR_BAYER_BG2BGR)
        img.Release()
        return cv2.flip(img_cv, -1)

    def process_next_frame(self, rc_override=False):
        """
        Pulls latest image from the camera and shows a scaled version of it on the window 'image Capture'.
        Draws rectangles for the perimeter [red], center [green], and the position of the last detected larva [blue].
        If ready_check is set at 1, the camera will write the image to image_file.

        :param rc_override: If True, it will write the image to image_file regardless of ready_check value. Default False.
        :return: None.
        """

        img = self.read_img()

        self.times.append(time.time())
        if len(self.times) > 100:
            self.times = self.times[-100:]
        fps = (len(self.times) + 1) / (self.times[-1] - self.times[0])

        # Colored indications for the perimeter [red], center [green],
        # and the position of the last detected larva [blue]
        copy = img.copy()
        display = cv2.resize(copy, (self.cam_w // 4, self.cam_h // 4))
        display[(self.mask_display == 1)] = colour_list[4]

        cv2.circle(
            display,
            (self.larva_x // 4, self.larva_y // 4),
            crop_size, colour_list[3], 1
        )
        cv2.rectangle(
            display,
            (stage_size // 4, stage_size // 4),
            ((self.cam_w - stage_size) // 4, (self.cam_h - stage_size) // 4),
            colour_list[1], 1
        )
        cv2.rectangle(
            display,
            ((stage_size + margin_size) // 4, (stage_size + margin_size) // 4),
            ((self.cam_w - stage_size - margin_size) // 4, (self.cam_h - stage_size - margin_size) // 4),
            colour_list[2], 1
        )
        cv2.putText(
            display,
            f'CURRENT SPEED: {fps:.2f} fps',
            (20, 40), cv2.FONT_HERSHEY_PLAIN,
            1, colour_list[1], thickness=1
        )

        cv2.imshow('Image Capture', display)

        if self.ready_check or rc_override:
            # writes image into file for the brain to read
            img.ravel()[self.mask_idx.ravel()] = 0
            img = img.reshape((self.cam_h, self.cam_w, 3))
            img_str = im.img2str(img, as_bw=True)
            self.publisher.send(f'lp search_img {img_str}', prt=False)

    def crop_img(self):
        """
        Takes image just around the larva position, currently a blue circle of radius 100.
        Crops off any parts that are hanging off agar area (green box on image).
        NOTE: THIS WILL NOT WORK IF CROP AREA IS OUTSIDE THE IMAGE AREA)

        :return: None. Sends out request for searching/parsing the cropped image.
        """

        img = self.read_img()
        img.ravel()[self.mask_idx.ravel()] = 0
        img = img.reshape((self.cam_h, self.cam_w, 3))
        crop_img = im.crop_img(img, self.larva_x, self.larva_y, crop_size)
        crop_str = im.img2str(crop_img, as_bw=True)
        self.publisher.send(f'lp search_crop_img {self.larva_x} {self.larva_y} {crop_str}')
        return img

    def set_larva_xy(self, x, y):
        self.larva_x = x
        self.larva_y = y

    def write_img(self, destination, save=False, ignore=False, as_bw=False):

        img = self.read_img()

        if ignore:
            img.ravel()[self.mask_idx.ravel()] = 0
            img = img.reshape((self.cam_h, self.cam_w, 3))
        if save is True and destination is not None:
            cv2.imwrite(destination, img)

        img_str = im.img2str(img, as_bw)
        print('Image updated!')
        return img, img_str

    def run(self):
        cv2.namedWindow('Image Capture')
        key = -1
        while True:
            key = cv2.waitKey(1)
            if key == 32:           # Spacebar
                # This restarts LarvaPicker process if it was stalled for any reason
                self.publisher.send(f'lp search_img')

                self.mask = np.ones((self.cam_h, self.cam_w, 3), int)
                self.mask[:stage_size, :] = [0, 0, 0]
                self.mask[self.cam_h - stage_size:, :] = [0, 0, 0]
                self.mask[:, :stage_size] = [0, 0, 0]
                self.mask[:, self.cam_w - stage_size:] = [0, 0, 0]
                self.mask_idx = (self.mask == 0)
                self.mask_display = np.zeros((self.cam_h // 4, self.cam_w // 4), int)

                self.ready_check = True

            elif key == 9:          # TAB
                # This halts LarvaPicker process and initiates recalibration protocol for robot
                self.publisher.send(f'lp hom_recal {self.larva_x} {self.larva_y}')
                self.ready_check = False

            elif key == ord('s'):
                self.publisher.send(f'lp pause_process')
                if self.ready_check:
                    self.ready_check = False
                else:
                    self.ready_check = True

            elif key == ord('f'):
                img, img_str = self.write_img(img_file, as_bw=True)
                self.publisher.send(f'lp deliver_to_larva {img_str}')
                self.ready_check = False

            elif key == ord('g'):
                # This halts LarvaPicker process and initiates feeding protocol for robot
                img, img_str = self.write_img(img_file, as_bw=True)
                self.publisher.send(f'lp hold_larva {img_str}')
                self.ready_check = False

            elif key == ord('h'):
                # This halts LarvaPicker process and initiates feeding protocol for robot
                img, img_str = self.write_img(img_file, as_bw=True)
                self.publisher.send(f'lp release_larva {img_str}')
                self.ready_check = False

            elif key == 27:         # ESC
                # This breaks out of the image capture loop and initiate cleanly ending all processes
                self.publisher.send(f'lp destroy')
                self.destroy()

            if self.poller.poll(self.subscriber):
                cmd, args = self.subscriber.recv()
                self.read(cmd, args)

            self.process_next_frame()


def main():
    cam = Grasshopper()
    cam.run()


if __name__ == '__main__':
    main()
