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
"""


import cv2
from docopt import docopt
import matplotlib.pyplot as plt
from tqdm import tqdm

from ...methods import images as im
from ...config.constants import colour_list
from ...config.__version__ import __version__
from ..utils.io import *


def flag(dataset_path):

    instar = 2
    h, w = 64, 64
    metadata = get_metadata(dataset_path)
    h_out, w_out = metadata['shape_y'], metadata['shape_x']
    seq_len = metadata['shape_t']
    print(f'Total number of frames in dataset: {seq_len}')

    testset_writer = TimestampedArrayWriter(
        None, dataset_path / 'st_flags' / 'test' / 'data.h5',
        (7,), np.float64,
        groupname=None,
        compression="gzip",
        compression_opts=5
    )
    targetset_writer = TimestampedArrayWriter(
        None, dataset_path / 'st_flags' / 'target' / 'data.h5',
        (1,), np.float64,
        groupname=None,
        compression="gzip",
        compression_opts=5
    )

    cv2.namedWindow('Video Track')
    flags = np.empty((0, 2))
    state_list = ['RUNNING', 'TURNING', 'STOPPED']
    state = 0
    t_prev = 0
    fn = -1
    fr = 1
    skip_rate = 30

    # actual video loop that interacts with user
    while True:
        fn += fr

        key = cv2.waitKey(30)
        if key == 27:
            print('KEYBOARD INTERRUPT ******** Exiting.')
            break
        elif key == 32:
            if state == 1:
                print('CHANGING STATE ******** Larva returning to run state.')
                state = 0
            else:
                print('CHANGING STATE ******** Larva now turning.')
                state = 1
        # elif key == ord('f'):
        #     if state == 2:
        #         print('CHANGING STATE ******** Larva returning to run state.')
        #         state = 0
        #     else:
        #         print('CHANGING STATE ******** Larva stopped!')
        #         state = 2
        elif key == ord('e'):
            print('USER INPUT ******** Paused. Press any key ONCE to resume.')
            cv2.waitKey()
            print('USER INPUT ******** Resuming.')
        elif key == ord('w'):
            print('USER INPUT ******** Increasing play speed.')
            if fr >= 30:
                print('WARNING ******** Cannot increase play speed any further!')
            else:
                fr *= 2
            print(f'Current play speed: {fr} frames per frame shown.')
        elif key == ord('s'):
            print('USER INPUT ******** Decreasing play speed.')
            if fr == 1:
                print('WARNING ******** Cannot decrease play speed any further!')
            else:
                fr = fr // 2
            print(f'Current play speed: {fr} frames per jump.')
        elif key == ord('a'):
            if fn - skip_rate <= 0:
                print('WARNING ******** ERROR!! Can\'t go back that far.')
            else:
                print(f'USER INPUT ******** '
                      f'Backing up to frame {fn - skip_rate} from frame {fn}')
                idx = np.where(flags[:, 0] >= fn - skip_rate)
                print(f'WARNING ******* Data deleted at frames: {flags[idx, 0]}')
                flags = np.delete(flags, idx, axis=0)
                fn = fn - skip_rate
            state = int(flags[-1, 1])
            fr = 1
        elif key == ord('d'):
            print(f'USER INPUT ******** '
                  f'Skipping to frame {fn + skip_rate} from frame {fn}')
            fn = fn + skip_rate

        if len(flags) > 0:
            while fn <= np.max(flags[:, 0]):
                idx = np.where(flags[:, 0] <= fn)
                print(f'WARNING ******* Duplicate frames deleted: {flags[idx, 0]}')
                flags = np.delete(flags, idx, axis=0)

        if fn >= seq_len:
            print('Finished processing video!')
            break
        frame = get_slice(dataset_path, fn)

        _frame = im.clean_img(frame)
        xy_list, cntr_list = im.get_contours(_frame, instar)
        n = len(xy_list)
        if n != 1:
            print('******* BAD FRAME. Skipping.')
            continue

        text = f'FRAME NUMBER: {fn} ({(fn*1000 // seq_len)/10}%)   PLAY RATE: x{fr}'
        fcopy = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
        cv2.putText(fcopy, text,
                    (w_out // 10, h_out // 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), thickness=2)
        cv2.putText(fcopy, state_list[state],
                    (w_out // 10, h_out // 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), thickness=2)
        cv2.circle(fcopy, (xy_list[0][0], xy_list[0][1]),
                   1, colour_list[state+1], -1)
        cv2.circle(fcopy, (xy_list[0][0], xy_list[0][1]),
                   60, colour_list[state+1], 3)
        scaled = cv2.resize(fcopy, (w_out, h_out))
        cv2.imshow('Video Track', scaled)

        flags = np.append(flags, [[fn, state]], axis=0).astype(int)

    cv2.destroyAllWindows()

    # using previous results to build the input/output for training dataset
    with h5py.File(dataset_path / '0' / 'coordinates.h5', 'r') as f:
        coordinates = np.zeros(f['data'].shape)
        f['data'].read_direct(coordinates)

    with h5py.File(dataset_path / '0' / 'pt_results.h5', 'r') as f:
        pt_results = np.zeros(f['data'].shape)
        f['data'].read_direct(pt_results)

    fn_prev = -1
    flag_results = np.ones((seq_len, 4)) * -1
    for (fn, state) in tqdm(flags, desc='Saving results', unit='flag'):
        for i in range(fn_prev+1, fn+1):

            t, cx, cy = coordinates[i, :3]
            tidx = np.where(pt_results[:, 0] == t)[0]
            if len(tidx) == 0:
                continue
            t, cx, cy, hx, hy, tx, ty = pt_results[tidx[0], :].astype(float)
            if t == 0:
                continue

            pt = np.array([
                t - t_prev,
                cx * 2 / w_out - 1.0, cy * 2 / h_out - 1.0,
                hx * 2 / w - 1.0, hy * 2 / h - 1.0,
                tx * 2 / w - 1.0, ty * 2 / h - 1.0
            ])
            testset_writer.append_data((t, pt))
            t_prev = t

            targetset_writer.append_data((t, np.array([float(state)])))
            flag_results[i, :] = np.array([cx, cy, state, i]).astype(int)
        fn_prev = fn
    testset_writer.close()
    targetset_writer.close()

    plt.figure()
    plt.title('st_flags')
    plt.scatter(
        flag_results[flag_results[:, 2] == 0, 0],
        flag_results[flag_results[:, 2] == 0, 1],
        color='g', marker='o', s=2, alpha=0.2
    )
    plt.scatter(
        flag_results[flag_results[:, 2] == 1, 0],
        flag_results[flag_results[:, 2] == 1, 1],
        color='r', marker='o', s=2, alpha=1.0
    )
    plt.savefig(str(dataset_path / 'st_flags' / 'flags.png'), dpi=300)

    return True


def main():
    args = docopt(__doc__, version=f'LarvaPicker {__version__}: StateTracker flagger')
    print(args, '\n')

    dataset_path = Path(args['--dataset_path'])
    index_path = Path(args['--index_path']) if args['--index_path'] else Path(__file__).parent / 'index.json'

    if not Path.is_dir(dataset_path / 'st_flags'):
        Path.mkdir(dataset_path / 'st_flags')
    if not Path.is_dir(dataset_path / 'st_flags' / 'target'):
        Path.mkdir(dataset_path / 'st_flags' / 'target')
    if not Path.is_dir(dataset_path / 'st_flags' / 'test'):
        Path.mkdir(dataset_path / 'st_flags' / 'test')

    ret = flag(dataset_path=dataset_path)

    if ret:
        if not Path.is_file(index_path):
            index = {}
        else:
            with open(index_path, 'r') as f:
                index = json.load(f)
        index[len(index)] = (dataset_path / 'st_flags').as_posix()
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=4)
        print(f'\nSaving index to: {index_path.as_posix()}'
              f'\nCurrent number of entries in index: {len(index)}')

    print('\n\n*** DONE!')


if __name__ == '__main__':
    main()
