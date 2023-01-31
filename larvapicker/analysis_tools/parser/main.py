"""
LarvaParser: analyzes video to track larva and split by individual
You can also run LarvaParser as a module with the same arguments:
    python -m LarvaRetriever.analysis_tools.parser --dataset_path=<dataset_path> [options]

Usage:
    main.py -h | --help
    main.py -v | --version
    main.py --dataset_path=<dataset_path> [options]

Options:
    -h --help                           show this message and exit.
    -v --version                        show version information and exit.
    --dataset_path=<dataset_path>       path to data directory to analyze.
    --n_larva=<n_larva>                 number of larva in experiment. [default: 1]
"""


import cv2
import time
import numpy as np
import json
from pathlib import Path
from docopt import docopt
from tqdm import tqdm

from .larva import Larva
from ..utils.io import get_metadata, get_slice
from ...config.__version__ import __version__
from ...methods import images as im


def run(
        dataset_path,
        n_larva
        ):

    print('Loading logs and prepping writers...')
    h, w = 64, 64
    metadata = get_metadata(dataset_path)
    instar = metadata['header']['instar']
    cam_h, cam_w, seq_len = metadata['shape_x'], metadata['shape_y'], metadata['shape_t']

    with open(dataset_path / 'log.json', 'r') as f:
        log = json.load(f)

    rec = cv2.VideoWriter(
        str(dataset_path / 'parsed.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30, (cam_w, cam_h), False
    )

    larva_list = [Larva(n, dataset_path, seq_len, (h, w)) for n in range(n_larva)]

    # cv2.namedWindow('LarvaTracker')
    key = -1
    start_time = time.time()
    print(f'\nStarting parsing frames ({seq_len} frames total)...\n')
    for t in tqdm(range(seq_len), desc='Parsing frames', unit='frames'):
        # key = cv2.waitKey(1)
        if key == 27:
            tqdm.write('KEYBOARD INTERRUPT ******** Exiting.')
            return False
        elif key == 32:
            tqdm.write('USER INPUT ******** Paused. Press any key ONCE to resume.')
            cv2.waitKey()
            tqdm.write('USER INPUT ******** Resuming.')

        # if t % 1000 == 0 and t > 0:
        #     tqdm.write(f'Frame #{t}\t{(t + 1) / (time.time() - start_time):.2f} fps')

        if log[str(t)]['status'] in ['INIT', 'PICKUP', 'DROPOFF', 'DELIVER', 'FEED']:
            tqdm.write(f'******* ROBOT ACTIVITY. Skipping analysis for frame #{t+1}.')
            for larva in larva_list:
                larva.write_crop(log[str(t)]['t'], None)
                larva.update_props(t, {})
            continue

        frame = get_slice(dataset_path, t)
        gray_frame = im.clean_img(frame)
        xy_list, cntr_list = im.get_contours(gray_frame, instar)
        n_t = len(xy_list)

        if n_t <= 0 or n_t > n_larva:
            tqdm.write(f'******* BAD FRAME. Found {n_t} larvae instead of {n_larva} in frame #{t+1}. Skipping analysis.')
            for larva in larva_list:
                larva.write_crop(log[str(t)]['t'], None)
                larva.update_props(t, {})
            continue
        elif n_t != n_larva:
            tqdm.write(f'Found {n_t} larvae instead of {n_larva} in frame #{t+1}.')
            pass

        if t > 0 and n_larva > 1:
            if log[str(t-1)]['status'] == 'DROPOFF':
                f_pickup = [f for f in range(t-10, t) if log[str(f)]['status'] == 'PICKUP']
                if len(f_pickup) > 0:
                    t_pickup = f_pickup[-1]
                    x_pickup, y_pickup = log[str(t_pickup)]['x'], log[str(t_pickup)]['y']
                    pickup_prob = -np.linalg.norm(
                        np.stack(
                            [[larva.props['x'] - x_pickup,
                              larva.props['y'] - y_pickup]
                             for larva in larva_list],
                            axis=0
                        ),
                        axis=-1
                    )
                    pickup_idx = np.argmax(pickup_prob)
                    larva_list[pickup_idx].log_activity(
                        t_pickup, t-1,
                        log[str(t_pickup)]['t'], log[str(t-1)]['t']
                    )
                    # larva_list[pickup_idx].update_props(
                    #     None, {'x': log[str(t-1)]['x'], 'y': log[str(t-1)]['y']})
                else:
                    pickup_idx = -1
            else:
                pickup_idx = -1

            id_prob = np.empty((n_t, n_larva))
            id_dict = {}
            for n in range(n_t):
                cx, cy = xy_list[n]
                area = cv2.contourArea(cntr_list[n])
                for m in range(n_larva):
                    id_prob[n, m] = -(
                        np.sqrt((larva_list[m].props['x'] - cx)**2
                                + (larva_list[m].props['y'] - cy)**2)
                        + 0.5 * np.abs(area - larva_list[m].props['area'])
                        # + 0.5 * np.abs(
                        #     np.sqrt((larva_list[m].props['x'] - cx)**2
                        #             + (larva_list[m].props['y'] - cy)**2)
                        #     - larva_list[m].props['mom']
                        # )
                    )
            # if pickup_idx >= 0:
            #     id_prob[:, pickup_idx] += -(cam_h + cam_w)/4

            for n in range(n_t):
                id_prob_max_h = np.argmax(np.max(id_prob, axis=1))
                id_prob_max_v = np.argmax(id_prob[id_prob_max_h, :])
                id_dict[int(id_prob_max_h)] = int(id_prob_max_v)
                id_prob[id_prob_max_h, :] = np.ones(n_larva) * -(cam_h + cam_w) * 2
                id_prob[:, id_prob_max_v] = np.ones(n_t) * -(cam_h + cam_w) * 2

        else:
            id_dict = {n: n for n in range(n_larva)}

        for n in range(n_t):
            nid = id_dict[n]
            cntr = np.array(cntr_list[n]).astype(np.int32)
            # cx, cy = xy_list[n].astype(np.int32)
            cx, cy = ((np.max(cntr, axis=0) + np.min(cntr, axis=0))/2).astype(np.int32)
            # cx, cy = np.array(cv2.minEnclosingCircle(cntr)[0]).astype(np.int32)
            area = cv2.contourArea(cntr)
            if cx <= 0 or cy <= 0 or cx >= cam_w or cy >= cam_h:
                crop = larva_list[nid].last_crop
                cx, cy = larva_list[nid].props['x'], larva_list[nid].props['y']
                print(cx, cy)
            else:
                temp = frame.copy()
                mask = np.zeros_like(temp)
                mask = cv2.drawContours(mask, [cntr], -1, 1, -1)
                mask = cv2.drawContours(mask, [cntr], -1, 1, 3)
                temp[mask == 0] = 0
                crop = im.crop_img(temp, cx, cy, w // 2)
                hj, wj = crop.shape
                if hj != h or wj != w:
                    print(hj, wj, cx, cy)
                    cx, cy = larva_list[n]
                    crop = im.crop_img(temp, cx, cy, w // 2)

            larva_list[nid].write_crop(log[str(t)]['t'], crop)
            larva_list[nid].update_props(
                t, {'t': log[str(t)]['t'], 'x': cx, 'y': cy, 'area': area}
            )

        missing = np.setdiff1d(np.arange(n_larva), np.array(list(id_dict.values())))
        for nid in missing:
            larva_list[nid].write_crop(log[str(t)]['t'], None)
            larva_list[nid].update_props(t, {})

        for n, larva in enumerate(larva_list):
            cv2.putText(gray_frame, f'LID#{n}',
                        (int(larva.props['x'] + 20), int(larva.props['y'] - 20)),
                        cv2.FONT_HERSHEY_PLAIN, 1, 255, thickness=1)

        cv2.putText(gray_frame, f'FRAME NUMBER: {t+1}',
                    (50, 90), cv2.FONT_HERSHEY_PLAIN,
                    1, 255, thickness=1)
        cv2.putText(gray_frame, f'CURRENT SPEED: {(t+1) / (time.time() - start_time):.2f} fps',
                    (50, 120), cv2.FONT_HERSHEY_PLAIN,
                    1, 255, thickness=1)
        rec.write(gray_frame)
        # scaled = cv2.resize(frame, (cam_w // 4, cam_h // 4))
        # cv2.imshow('LarvaTracker', scaled)

    for larva in larva_list:
        larva.save()
    rec.release()
    # cv2.destroyAllWindows()

    print(f'\nFinished processing {seq_len}. '
          f'Total time elapsed: {time.time()-start_time:.0f}s')

    return True


def main():
    args = docopt(__doc__, version=f'LarvaPicker {__version__}: LarvaParser')
    print(args, '\n')

    ret = run(
        dataset_path=Path(args['--dataset_path']),
        n_larva=int(args['--n_larva'])
    )

    if ret:
        print('\n\n*** DONE!')


if __name__ == '__main__':
    main()
