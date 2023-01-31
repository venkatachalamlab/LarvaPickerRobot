"""
StateTracker: classify larva behavior state with a recurrent neural network.
You can also run StateTracker as a module with the same arguments:
    state_tracker --dataset_path=<dataset_path> [options]

Usage:
    main.py -h | --help
    main.py -v | --version
    main.py --dataset_path=<dataset_path> [options]

Options:
    -h --help                           show this message and exit.
    -v --version                        show version information and exit.
    --dataset_path=<dataset_path>       path to data directory to analyze.
    --checkpoint_path=<checkpoint_path> path to training checkpoint file.
    --dropout_rate=<dropout_rate>       fraction of intermediate data to drop out during training. [default: 0]
    --batch_size=<batch_size>           batch size. [default: 1]
"""


import time
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from docopt import docopt
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...config.constants import colour_list
from ...config.__version__ import __version__
from ..utils.io import *
from ..utils.streamers import DataStreamer
from .model import StateTracker


def state_tracker(
        dataset_path,
        n_channels_in=7,
        n_channels_out=1,
        init_nodes=8,
        dropout_rate=0.,
        dev=torch.device('cpu'),
        batch_size=100,
        state_dict=None):

    h, w = 64, 64
    h_out, w_out = 1024, 1024

    print('\n\nCompiling model...')
    model = StateTracker(
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        n_chunks=1,
        init_nodes=init_nodes,
        dropout_rate=dropout_rate
    ).to(dev)
    print('Loading model weights from previous checkpoint...')
    model.load_state_dict(state_dict)

    print('\nLoading data streamer...')
    data_streamer = DataStreamer(dataset_path, dev, file_name='pt_results.h5')
    data_loader = DataLoader(
        data_streamer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    results = np.zeros((len(data_streamer)))

    model.eval()
    for i, batch in enumerate(tqdm(data_loader, desc='Analyzing behavior', unit='batches')):
        nidx = torch.nonzero(batch[:, 0], as_tuple=True)[0]
        _batch = batch[nidx, :]
        input_tensor = torch.stack(
            (
                _batch[1:, 0] - _batch[:-1, 0],
                _batch[1:, 1] * 2 / w_out - 1.0,
                _batch[1:, 2] * 2 / h_out - 1.0,
                _batch[1:, 3] * 2 / w - 1.0,
                _batch[1:, 4] * 2 / h - 1.0,
                _batch[1:, 5] * 2 / w - 1.0,
                _batch[1:, 6] * 2 / h - 1.0,
            ),
            dim=-1
        ).unsqueeze(0).to(torch.float32)
        pred = model(input_tensor)
        # print(torch.max(pred))
        _pred = pred.detach().cpu().numpy().flatten()
        results[nidx.detach().cpu().numpy()[1:] + i * batch_size
                ] = np.where(_pred > 0.1, 1, 0).astype(int)

    save_data(results, dataset_path / 'st_results.h5')

    with h5py.File(dataset_path / "coordinates.h5", 'r') as f:
        coordinates = np.zeros(f["data"].shape)
        f["data"].read_direct(coordinates)
    coordinates = np.array(coordinates)[(data_streamer.times).astype(int), :]
    slice_x = coordinates[:, 1].flatten()
    slice_y = coordinates[:, 2].flatten()
    plt.figure(0, figsize=[16, 12], dpi=400)
    plt.title('StateTracker Result')
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.scatter(slice_x[(results == 0)], slice_y[(results == 0)],
                color='g', marker='o', s=1, alpha=0.33, label='Run')
    plt.scatter(slice_x[(results == 1)], slice_y[(results == 1)],
                color='darkorange', marker='o', s=1, alpha=0.66, label='Turn')
    plt.legend(loc='upper left', bbox_to_anchor=(.8, 0.9))
    plt.savefig((dataset_path / 'st_results.png').as_posix())

    return True


def main():
    args = docopt(__doc__, version=f'LarvaPicker {__version__}: StateTracker')
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

    ret = state_tracker(
        dataset_path=Path(args['--dataset_path']),
        n_channels_in=7,
        n_channels_out=1,
        init_nodes=16,
        dropout_rate=float(args['--dropout_rate']),
        dev=dev,
        batch_size=int(args['--batch_size']),
        state_dict=checkpoint['state_dict']
    )

    if ret:
        print('\n\n*** DONE!')


if __name__ == '__main__':
    main()
