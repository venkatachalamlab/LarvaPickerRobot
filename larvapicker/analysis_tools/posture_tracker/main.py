"""
PostureTracker: track larva postures via a modified U-Net.
You can also run PostureTracker as a module with the same arguments:
    posture_tracker --dataset_path=<dataset_path> [options]

Usage:
    main.py -h | --help
    main.py -v | --version
    main.py --dataset_path=<dataset_path> [options]

Options:
    -h --help                           show this message and exit.
    -v --version                        show version information and exit.
    --dataset_path=<dataset_path>       path to data directory to analyze.
    --checkpoint_path=<checkpoint_path> path to training checkpoint file.
    --init_nodes=<init_nodes>           number of kernels for the first layer. [default: 16]
    --batch_size=<batch_size>           batch size. [default: 1]
"""


import time
import json
import cv2
import numpy as np
import scipy.spatial as sp
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from docopt import docopt
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...config.constants import colour_list
from ...config.__version__ import __version__
from ...methods.images import clean_img, get_contours, interpolate_contour
from ..utils.io import *
from ..utils.streamers import DataStreamer
from .model import PostureTracker


def posture_tracker(
        dataset_path,
        img_shape=(128, 128),
        n_channels_in=1,
        n_channels_out=2,
        init_nodes=16,
        dev=torch.device('cpu'),
        batch_size=100,
        state_dict=None
        ):

    metadata = get_metadata(dataset_path.parent)
    instar = int(metadata['instar'])
    with open(dataset_path.parent / 'log.json', 'r') as f:
        log = json.load(f)
    with h5py.File(dataset_path / "coordinates.h5", 'r') as f:
        coordinates = np.zeros(f["data"].shape)
        f["data"].read_direct(coordinates)
    coordinates = np.array(coordinates)

    print('\n\nCompiling model...', coordinates.shape)
    model = PostureTracker(
        img_shape=img_shape,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        init_nodes=init_nodes,
        kernel=(3, 3),
        padding=1,
        pool_kernel=2
    ).to(dev)
    print('Loading model weights from previous checkpoint...')
    model.load_state_dict(state_dict)

    print('\nLoading data streamer...')
    data_streamer = DataStreamer(dataset_path, dev)
    data_loader = DataLoader(
        data_streamer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    results = np.zeros((len(data_streamer), 7))
    rec = cv2.VideoWriter(
        str(dataset_path / 'pt_results.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30, img_shape, True
    )
    rec_pred = cv2.VideoWriter(
        str(dataset_path / 'pt_pred.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30, img_shape, True
    )

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc='Analyzing postures', unit='batches')):
            batch = batch.unsqueeze(0).unsqueeze(2)
            batch = batch.to(torch.float16) / torch.max(batch)
            pred = model(batch)
            pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy()
            pred[:, :, 0, ...] = (pred[:, :, 0, ...] / np.max(pred[:, :, 0, ...]) * 255)
            pred[:, :, 1, ...] = (pred[:, :, 1, ...] / np.max(pred[:, :, 1, ...]) * 255)

            for j in range(pred.shape[1]):
                fn = i*batch_size + j

                if fn-1 < 0:
                    pass
                elif log[str(fn)]['status'] in ['INIT', 'PICKUP', 'DROPOFF', 'DELIVER', 'FEED']:
                    continue
                elif coordinates[fn, 1] <= 0 or coordinates[fn, 2] <= 0:
                    tqdm.write(f'*** Missing coordinates {fn}: {coordinates[fn, 1:3]}')
                    continue

                img_head = pred[0, j, 0, ...].astype(np.uint8)
                if j > 0:
                    tx_last = int(results[fn-1, 5])
                    ty_last = int(results[fn-1, 6])
                    if tx_last > 2 and ty_last > 2:
                        img_head[ty_last-2:ty_last+3, tx_last-2:tx_last+3] = np.zeros((5, 5))
                img_head_blur = cv2.blur(img_head, (7, 7))
                _, _, _, maxLoc = cv2.minMaxLoc(img_head_blur)
                hx, hy = maxLoc

                img_tail = pred[0, j, 1, ...].astype(np.uint8)
                if hx > 2 and hy > 2:
                    img_tail[hy-2:hy+3, hx-2:hx+3] = np.zeros((5, 5))
                img_tail_blur = cv2.blur(img_tail, (7, 7))
                _, _, _, maxLoc = cv2.minMaxLoc(img_tail_blur)
                tx, ty = maxLoc

                # data = compare_red_green(pred[0, j, 0, ...], pred[0, j, 1, ...])
                data = compare_red_green(img_head_blur, img_tail_blur)
                rec_pred.write(data[:, :, ::-1])

                if hx <= 0 or hy <= 0 or tx <= 0 or ty <= 0:
                    tqdm.write(f'*** No head/tail found {fn}: ({hx}, {hy})/({tx}, {ty})')
                    continue

                _frame = clean_img((batch[0, j, 0, ...].cpu().numpy() * 255).astype(np.uint8))
                xy_list, cntr_list = get_contours(_frame, instar)
                if cntr_list is None:
                    tqdm.write(f'*** Bad contour {fn}')
                    continue
                contour = interpolate_contour(np.array(cntr_list[0]), 36)

                hdist = sp.distance.cdist(contour, np.array([[hx, hy]]))
                hidx = np.argmin(hdist.flatten())
                contour = np.append(contour[hidx:, :], contour[:hidx, :], axis=0)

                tdist = sp.distance.cdist(contour[2:34, :], np.array([[tx, ty]]))
                tidx = np.argmin(tdist.flatten()) + 2

                hcntr, tcntr = contour[:tidx, :], np.flip(contour[tidx:, :], axis=0)
                if len(hcntr) <= 2 or len(tcntr) <= 2:
                    tqdm.write(f'*** Insufficient contour length {fn}: {len(hcntr)}, {len(tcntr)}')
                    continue

                hcntr, tcntr = interpolate_contour(hcntr, 19), interpolate_contour(tcntr, 19)
                midspine = np.around([(hcntr[9, 0] + tcntr[9, 0]) / 2, (hcntr[9, 1] + tcntr[9, 1]) / 2])
                if cv2.pointPolygonTest(contour.astype(int), (int(midspine[0]), int(midspine[1])), False) >= 0:
                    coordinates[fn, 1] += (midspine[0] - img_shape[0]//2)
                    coordinates[fn, 2] += (midspine[1] - img_shape[1]//2)
                else:
                    tqdm.write(f'*** Spine misalinged {fn}')
                hx, hy = int(hcntr[0, 0]), int(hcntr[0, 1])
                tx, ty = int(tcntr[-1, 0]), int(tcntr[-1, 1])

                if j > 0:
                    body_length = (np.linalg.norm([hx - midspine[0], hy - midspine[1]]) +
                                   np.linalg.norm([tx - midspine[0], ty - midspine[1]]))
                    if body_length < 3:
                        tqdm.write(f'*** Low body length {fn}: {body_length:.3f}')
                        continue
                    elif body_length > 40:
                        tqdm.write(f'*** High body length {fn}: {body_length:.3f}')
                        continue

                    if not np.any(results[fn-1, :] == 0):
                        diff_hth = np.linalg.norm([hx - results[fn-1, 3],
                                                   hy - results[fn-1, 4]])
                        diff_htt = np.linalg.norm([hx - results[fn-1, 5],
                                                   hy - results[fn-1, 6]])
                        diff_tth = np.linalg.norm([tx - results[fn-1, 3],
                                                   ty - results[fn-1, 4]])
                        diff_ttt = np.linalg.norm([tx - results[fn-1, 5],
                                                   ty - results[fn-1, 6]])
                        if diff_hth - diff_htt > 0.7 * body_length and diff_ttt - diff_tth > 0.7 * body_length:
                            tqdm.write(f'*** Head-Tail flip with hth {diff_hth:.3f} and htt {diff_htt:.3f} '
                                       f'with body length {body_length:.3f} at {fn}')
                            continue

                results[fn, :] = np.array(
                    [log[str(fn)]['t'],
                     coordinates[fn, 1],
                     coordinates[fn, 2],
                     hx, hy,
                     tx, ty],
                    dtype=float
                )

                _frame = cv2.cvtColor(_frame, cv2.COLOR_GRAY2BGR)
                cv2.circle(_frame, (hx, hy), 1, colour_list[2], -1)
                cv2.circle(_frame, (int(midspine[0]), int(midspine[1])), 1, colour_list[3], -1)
                cv2.circle(_frame, (tx, ty), 1, colour_list[1], -1)
                rec.write(np.array(_frame, dtype=np.uint8))

        print(f'\n\n*** Finished analyzing dataset!'
              f'\nNumber of frames successfully analyzed: {len(np.where(results[:, 0] != 0)[0]) + 1}'
              f'\nNumber of frames discarded: {len(np.where(results[:, 0] == 0)[0]) - 1}'
              f'\nPercentage of raw data retained:'
              f'\t{(1 - len(np.where(results[:, 0] == 0)[0]) / len(data_streamer)) * 100:2.4f}%'
              f'\nTotal time elapsed: {time.time()-start_time:.0f}s')

        # save_data(coordinates, dataset_path / 'coordinates.h5')
        tlist = data_streamer.times[np.where(results[:, 0] != 0)[0]]
        results = results[np.where(results[:, 0] != 0)[0], :]
        save_data(results, dataset_path / 'pt_results.h5', tlist=tlist)
        rec.release()
        rec_pred.release()

    return True


def main():
    args = docopt(__doc__, version=f'LarvaPicker {__version__}: PostureTracker')
    print(args, '\n')

    if torch.cuda.is_available():
        # Moving to GPU
        print('\n*** GPU available!\n')
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    checkpoint_path = Path(args['--checkpoint_path']) \
        if args['--checkpoint_path'] else Path(__file__).parent / 'checkpoint.pt'
    if Path.is_file(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
    else:
        print('\nCannot find checkpoint for model weights! Exiting...')
        exit()

    ret = posture_tracker(
            dataset_path=Path(args['--dataset_path']),
            img_shape=(64, 64),
            n_channels_in=1,
            n_channels_out=2,
            init_nodes=int(args['--init_nodes']),
            dev=dev,
            batch_size=int(args['--batch_size']),
            state_dict=checkpoint['state_dict']
    )

    if ret:
        print('\n\n*** DONE!')


if __name__ == '__main__':
    main()
