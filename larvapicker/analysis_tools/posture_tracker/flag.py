"""
PostureFlagger: flag larva postures for training PostureTracker's core neural network.

Usage:
    flag.py -h | --help
    flag.py -v | --version
    flag.py --dataset_path=<dataset_path> [options]

Options:
    -h --help                           show this message and exit.
    -v --version                        show version information and exit.
    --dataset_path=<dataset_path>       path to data directory to analyze.
    --index_path=<index_path>           path to index of data directories with flagged data.
    --t_start=<t_start>                 frame to start flagging
    --t_end=<t_end>                     frame to end flagging
"""


import cv2
from docopt import docopt
import scipy.spatial as sp

from ...methods import images as im
from ...config.constants import colour_list
from ...config.__version__ import __version__
from ..utils.io import *


def flag(dataset_path, t_start, t_end):

    instar = 2
    h, w = 64, 64
    ftotal = 1000

    log_file = dataset_path / 'log.json'
    with open(log_file, 'r') as f:
        log = json.load(f)

    if t_start is None:
        i = 0
        act_idx = [k for k, v in log.items() if v['status'] != '1']
        while True:
            count = int(act_idx[i + 1]) - int(act_idx[i])
            if count >= 6000:
                t_start = int(act_idx[i]) + 2001
                t_end = t_start + 10*ftotal
                print(f'Setting starting frame number at: {t_start}\n'
                      f'Setting ending frame number at: {t_end}')
                break
            i += 1
            if i >= len(act_idx) - 1:
                print('No sufficient candidate found.')
                t_start = act_idx[0]
                t_end = act_idx[1]
                break
    elif t_end is None:
        t_end = t_start + 10*ftotal

    targetset_writer = TimestampedArrayWriter(
        None, dataset_path / 'pt_flags' / 'target' / 'data.h5',
        (2, w, h),
        np.uint8,
        groupname=None,
        compression="gzip",
        compression_opts=5
    )
    testset_writer = TimestampedArrayWriter(
        None, dataset_path / 'pt_flags' / 'test' / 'data.h5',
        (w, h),
        np.uint8,
        groupname=None,
        compression="gzip",
        compression_opts=5
    )
    posture_rec = cv2.VideoWriter(
        str(dataset_path / 'pt_flags' / 'movie.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        10, (w, h), True
    )
    head_rec = cv2.VideoWriter(
        str(dataset_path / 'pt_flags' / 'head.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        10, (w, h), False
    )
    tail_rec = cv2.VideoWriter(
        str(dataset_path / 'pt_flags' / 'tail.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        10, (w, h), False
    )
    coordinates = np.empty((0, 7))

    fcount = 0
    hx, hy, tx, ty = 0, 0, 0, 0
    cv2.namedWindow('PostureFlagger')
    for i, t in enumerate(range(t_start, t_end, 5)):
        key = cv2.waitKey(1)
        if key == 27:       # esc
            print('KEYBOARD INTERRUPT ******** Exiting.')
            return False
        elif key == 32:     # spacebar
            print('USER INPUT ******** Paused. Press any key ONCE to resume.')
            cv2.waitKey()
            print('USER INPUT ******** Resuming.')

        frame = get_slice(dataset_path / '0', t)
        if frame is None or fcount >= ftotal or t >= t_end:
            print(f'Finished processing video with {fcount + 1} frames encoded.')
            break

        _frame = im.clean_img(frame)
        xy_list, cntr_list = im.get_contours(_frame, instar)
        n = len(xy_list)
        if n != 1:
            print('******* BAD FRAME. Skipping.')
            continue

        x, y = xy_list[0][0], xy_list[0][1]
        crop_contour = cntr_list - np.array([[x - w//2, y - h//2]])

        fcount += 1
        print("Select the head of the larvae.")
        im.mouseX, im.mouseY = 4*hx, 4*hy
        while True:
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            cv2.setMouseCallback('PostureFlagger', im.onMouse)
            head = np.array([im.mouseX // 4, im.mouseY // 4])

            fcopy = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
            cv2.circle(fcopy, (im.mouseX // 4, im.mouseY // 4), 1, colour_list[1], -1)
            scaled = cv2.resize(fcopy, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
            text = f'FN: {t} ({fcount}/{ftotal})'
            cv2.putText(scaled, text, (2, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)
            cv2.imshow('PostureFlagger', scaled)

        hdist = sp.distance.cdist(crop_contour[0], np.array([head]))
        hidx = np.argmin(hdist.flatten())
        hx, hy = crop_contour[0][hidx]

        head_crop = np.zeros_like(frame)
        head_crop[hy-2:hy+3, hx-2:hx+3] = 255
        head_crop = cv2.GaussianBlur(head_crop, (7, 7), 1.5)
        # head_crop = cv2.drawContours(head_crop, crop_contour, -1, 20, 1)
        head_crop[hy-1:hy+2, hx-1:hx+2] = 255
        head_rec.write(np.uint8(head_crop))

        print("Select the tail of the larvae.")
        tail = coordinates[i, 5:7]
        im.mouseX, im.mouseY = 4*tx, 4*ty
        while True:
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            cv2.setMouseCallback('PostureFlagger', im.onMouse)
            tail = np.array([im.mouseX // 4, im.mouseY // 4])

            fcopy = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
            cv2.circle(fcopy, (hx, hy), 1, colour_list[1], -1)
            cv2.circle(fcopy, (im.mouseX // 4, im.mouseY // 4), 1, colour_list[2], -1)
            scaled = cv2.resize(fcopy, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
            text = f'FN: {t} ({fcount}/{ftotal})'
            cv2.putText(scaled, text, (2, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)
            cv2.imshow('PostureFlagger', scaled)

        tdist = sp.distance.cdist(crop_contour[0], np.array([tail]))
        tidx = np.argmin(tdist.flatten())
        tx, ty = crop_contour[0][tidx]

        tail_crop = np.zeros_like(frame)
        tail_crop[ty-2:ty+3, tx-2:tx+3] = 255
        tail_crop = cv2.GaussianBlur(tail_crop, (7, 7), 1.5)
        # tail_crop = cv2.drawContours(tail_crop, crop_contour, -1, 20, 1)
        tail_crop[ty-1:ty+2, tx-1:tx+2] = 255
        tail_rec.write(np.uint8(tail_crop))

        _t = log[str(t)]['t']
        target = np.append(head_crop[np.newaxis, ...], tail_crop[np.newaxis, ...], axis=0)
        targetset_writer.append_data((_t, target))
        testset_writer.append_data((_t, frame[np.newaxis, ...]))
        coordinates = np.append(coordinates, np.array([[_t, x, y, *head, *tail]]), axis=0)
        posture_rec.write(fcopy)

    save_data(coordinates, dataset_path / 'pt_flags' / 'coordinates.h5')
    targetset_writer.close()
    testset_writer.close()
    posture_rec.release()
    head_rec.release()
    tail_rec.release()

    return True


def main():
    args = docopt(__doc__, version=f'LarvaPicker {__version__}: PostureTracker flagger')
    print(args, '\n')

    dataset_path = Path(args['--dataset_path'])
    index_path = Path(args['--index_path']) if args['--index_path'] else Path(__file__).parent / 'index.json'

    if not Path.is_dir(dataset_path / 'pt_flags'):
        Path.mkdir(dataset_path / 'pt_flags')
    if not Path.is_dir(dataset_path / 'pt_flags' / 'target'):
        Path.mkdir(dataset_path / 'pt_flags' / 'target')
    if not Path.is_dir(dataset_path / 'pt_flags' / 'test'):
        Path.mkdir(dataset_path / 'pt_flags' / 'test')

    ret = flag(dataset_path=dataset_path,
               t_start=int(args['--t_start']) if args['--t_start'] else None,
               t_end=int(args['--t_end']) if args['--t_end'] else None)

    if ret:
        if not Path.is_file(index_path):
            index = {}
        else:
            with open(index_path, 'r') as f:
                index = json.load(f)
        index[len(index)] = (dataset_path / 'pt_flags').as_posix()
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=4)
        print(f'\nSaving index to: {index_path.as_posix()}'
              f'\nCurrent number of entries in index: {len(index)}')

    print('\n\n*** DONE!')


if __name__ == '__main__':
    main()
