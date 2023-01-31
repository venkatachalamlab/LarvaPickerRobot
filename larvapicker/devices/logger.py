"""
This handles the "memory" of the LarvaPicker, aka the data storage.
This saves pertinent information about the experiment as .npy and .txt as well as saving a video of the entire duration.
* Only configured for single larva experiments.

Usage:
    LarvaLogger.py


Server connections:
    --inbound=HOST:PORT       Connect for inbound messages from the LarvaPicker.
                              [default: localhost:5000]
"""

import numpy as np
import cv2
import zmq
import json

from ..analysis_tools.utils.io import TimestampedArrayWriter
from ..config.constants import *
from ..methods import images as im
from ..utils.subscriber import Subscriber
from ..utils.poller import Poller


class LarvaLogger:
    def __init__(self):
        context = zmq.Context()
        self.subscriber = Subscriber(name='log',
                                     port='5000',
                                     context=context)
        self.poller = Poller()
        self.poller.register(self.subscriber)
        if not dataset_path.exists():
            dataset_path.mkdir()

        # Manually input instar of the larvae
        self.instar = int(input('Select instar: 1, 2, or 3? '))
        while self.instar not in [1, 2, 3, 4]:
            self.instar = int(input('******* ERROR!'
                                    '\nPlease select proper instar: 1, 2, or 3? '))

        notes = input('Any additional notes?'
                      '\n[B]aseline / [T]hermotaxis / [N]icotine ')

        # Sets up data writer, all frames at full resolution with time stamps
        self.writer = TimestampedArrayWriter(
            None, dataset_file,
            (w_out, h_out),
            np.uint8,
            groupname=None,
            compression="lzf"
        )

        # Sets up video writer object in mp4 format, fps=10, full resolution
        self.fn = 0
        fr = 10
        self.rec = cv2.VideoWriter(
            str(movie),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fr, (w_out, h_out), False
        )
        print(f'Video recorder opened, recording at fps of {fr}')

        self.metadata = {
            'header': {'date': date, 'instar': self.instar, 'notes': notes},
            'shape_x': w_out, 'shape_y': h_out
        }
        self.logger = {}

        self.prev_frame = np.zeros((cam_h, cam_w))
        self.prev_time = 0
        self.active_time = 0
        self.ct_offset = 0
        self.times = []
        print('Opening memory bank. Now available for long term memory storage.')

    def destroy(self, counts):
        """
        Closes and releases all handles and exits process
        """

        self.metadata['shape_t'] = self.fn
        self.metadata['footer'] = {
            'total': self.prev_time,
            'active': self.active_time,
            '%': (self.active_time/self.prev_time)*100
        }
        try:
            self.metadata['pickup'] = {
                'total': int(counts[0] + 2*counts[1] + 3*counts[2] + counts[5]),
                'first': int(counts[0]),
                'second': int(counts[1]),
                'third': int(counts[2]),
                'fail': int(counts[5]),
                '%': counts[5] / (counts[0] + 2*counts[1] + 3*counts[2] + counts[5] + 1e-6) * 100
            }
            self.metadata['dropoff'] = {
                'total': int(counts[3] + 2*counts[4] + counts[6]),
                'first': int(counts[3]),
                'second': int(counts[4]),
                'fail': int(counts[6]),
                '%': counts[6] / (counts[3] + 2*counts[4] + counts[6] + 1e-6) * 100
            }
        except:
            pass

        print('Releasing recorder and saving files.')
        self.writer.close()
        self.rec.release()
        with open(dataset_path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=4)
        with open(log_file, 'w') as f:
            json.dump(self.logger, f, indent=4)

        exit()

    def read(self, cmd, args):
        if cmd == 'destroy':
            self.destroy(np.array(args, dtype=int))

        elif cmd == 'write_to_memory':
            if len(args) > 4:
                self.write_to_memory(
                    float(args[0]), float(args[1]),
                    str(args[2]),
                    float(args[3]),
                    ' '.join(args[4:])
                )
            else:
                self.write_to_memory(
                    float(args[0]), float(args[1]),
                    str(args[2]),
                    float(args[3]),
                    None
                )

        else:
            print('******* ERROR: UNRECOGNIZED REQUEST! Skipping.')

    def write_to_memory(self,
                        x, y,
                        status,
                        current_time,
                        img_str):

        if self.fn > 0:
            if self.logger[self.fn-1]['t'] > current_time + self.ct_offset:
                self.ct_offset = self.logger[self.fn]['t']
                print(f'******* LARVAPICKER RESTARTED.'
                      f'\nOffsetting incoming time by: {self.ct_offset}')
        current_time += self.ct_offset

        self.times.append(current_time)
        if len(self.times) > 100:
            self.times = self.times[-100:]
        fps = len(self.times) / (self.times[-1] - self.times[0])
        print(f'Recording frame {self.fn+1} @ {current_time} ({fps}fps)')

        if img_str is not None:
            img = im.str2img(img_str, as_bw=True)
            frame = img[stage_size+margin_size:cam_h-stage_size-margin_size,
                        stage_size+margin_size:cam_w-stage_size-margin_size]
            frame = cv2.resize(frame, (w_out, h_out))
            self.prev_frame = frame
        else:
            frame = self.prev_frame
        self.writer.append_data((self.fn, frame))
        self.rec.write(frame)

        if status in ['INIT', 'PICKUP', 'DROPOFF', 'DELIVER', 'FEED']:
            self.active_time += current_time - self.prev_time
        x, y = (x-stage_size-margin_size) * w_out/cam_w, (y-stage_size-margin_size) * h_out/cam_h
        self.logger[self.fn] = {'x': int(x), 'y': int(y), 't': current_time, 'status': status}

        self.prev_time = current_time
        self.fn += 1

    def run(self):
        while True:
            if self.poller.poll(self.subscriber):
                cmd, args = self.subscriber.recv(prt=False)
                self.read(cmd, args)


def main():
    memory = LarvaLogger()
    memory.run()


if __name__ == '__main__':
    main()
