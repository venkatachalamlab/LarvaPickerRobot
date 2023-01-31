import cv2
import numpy as np
from pathlib import Path

from ..utils.io import TimestampedArrayWriter, save_data


class Larva:
    def __init__(self, idx, dataset_path, seq_len, crop_shape):
        self.props = {
            'id': idx,
            't': 0, 'x': 0, 'y': 0, 'area': -1,
            'mom': 0
        }
        self.dataset_path = dataset_path / str(self.props['id'])
        if not Path.is_dir(self.dataset_path):
            Path.mkdir(self.dataset_path)

        self.df = np.zeros((seq_len, 3))
        self.activity_log = np.empty((0, 4))

        self.last_crop = np.zeros(crop_shape)
        self.writer = TimestampedArrayWriter(
            None, self.dataset_path / 'data.h5',
            crop_shape,
            np.uint8,
            groupname=None,
            compression="gzip",
            compression_opts=5
        )
        self.recorder = cv2.VideoWriter(
            str(self.dataset_path / 'movie.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            30, crop_shape, False
        )

    def update_props(self, tidx, props):
        row = [-1, -1, -1]
        for k, v in props.items():
            if k == 't':
                row[0] = v
            elif k == 'x':
                row[1] = v
            elif k == 'y':
                row[2] = v
            if -1 not in row:
                self.props['mom'] = (
                    0.8 * self.props['mom']
                    + 0.2 * np.sqrt(
                        (row[1]-self.props['x'])**2
                        + (row[2]-self.props['y'])**2
                    )
                )

            # if k == 'area':
            #     self.props[k] = 0.99 * self.props[k] + 0.01 * float(v)
            # else:
            self.props[k] = float(v)

        if tidx is not None:
            if -1 not in row:
                self.df[tidx] = np.array(row)
            else:
                self.df[tidx] = self.df[tidx-1]

    def log_activity(self, tidx_pickup, tidx_dropoff, t_pickup, t_dropoff):
        self.activity_log = np.append(
            self.activity_log,
            [[tidx_pickup, tidx_dropoff, float(t_pickup), float(t_dropoff)]],
            axis=0
        )

    def write_crop(self, t, crop=None):
        if crop is None:
            crop = self.last_crop
        else:
            self.last_crop = crop
        self.writer.append_data((t, crop[np.newaxis, ...]))
        self.recorder.write(crop)

    def save(self):
        save_data(
            self.df,
            self.dataset_path / 'coordinates.h5'
        )
        save_data(
            self.activity_log,
            self.dataset_path / 'lp_log.h5'
        )
        self.writer.close()
        self.recorder.release()
