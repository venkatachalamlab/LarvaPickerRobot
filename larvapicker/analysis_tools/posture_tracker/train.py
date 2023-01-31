"""
PostureTracker trainer: trains PostureTracker's core neural network with flagged data from PostureFlagger.

Usage:
    train.py -h | --help
    train.py -v | --version
    train.py [options]

Options:
    -h --help                           show this message and exit.
    -v --version                        show version information and exit.
    --index_path=<index_path>           path to index of data directories with flagged data.
    --checkpoint_path=<checkpoint_path> path to training checkpoint file.
    --init_nodes=<init_nodes>           number of kernels for the first layer. [default: 16]
    --n_epoch=<n_epoch>                 number of training epochs. [default: 10]
    --lr_init=<lr_init>                 initial learning rate. [default: 0.1]
    --batch_size=<batch_size>           batch size. [default: 1]
    --val_split=<val_split>             fraction of data to use for validation. [default: 0]
    --override=<override>               override existing training checkpoint. [default: False]
"""


import time
import cv2
import datetime
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.optim as optim

from docopt import docopt
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ...config.constants import date
from ...config.__version__ import __version__
from ..utils.io import *
from ..utils.streamers import MultiDataStreamer
from .model import PostureTracker


def train_model(
        index_path=Path('./index.json'),
        img_shape=(128, 128),
        n_channels_in=1,
        n_channels_out=2,
        init_nodes=16,
        dev=torch.device('cpu'),
        n_epoch=10,
        lr_init=0.1,
        batch_size=1,
        validation_split=0.,
        state_dict=None,
        opt_dict=None,
        loss_list=None,
        ):

    print('\n\nCompiling model...')
    model = PostureTracker(
        img_shape=img_shape,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        init_nodes=init_nodes,
        kernel=(3, 3),
        padding=1,
        pool_kernel=2
    ).to(dev)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=lr_init, eps=1e-6)
    # if dev == 'cuda':
    scaler = torch.cuda.amp.GradScaler()

    if state_dict is not None:
        print('Loading model weights from previous checkpoint...')
        model.load_state_dict(state_dict)

    print('\n\nLoading data streamer...')
    data_streamer = MultiDataStreamer(index_path, dev)
    n_data = len(data_streamer)
    n_val = int(np.floor(validation_split * n_data))
    n_train = n_data - n_val
    if n_val > 0:
        data_streamer, stream_val = random_split(data_streamer, [n_train, n_val])
        validation_input = DataLoader(
            stream_val,
            batch_size=n_val,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    training_input = DataLoader(
        data_streamer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # training loop here
    model.train()
    pbar = tqdm(range(n_epoch), desc='Training', unit='epochs')
    for epoch in pbar:

        with torch.no_grad():
            loss_list = []

            if epoch > 0 and epoch % 4 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                optimizer = optim.Adadelta(model.parameters(), lr=lr_init, eps=1e-6)
                scaler = torch.cuda.amp.GradScaler()
                new_lr = optimizer.param_groups[0]['lr']
                tqdm.write(f'Epoch: {epoch}\t\t'
                           f'Loss: {mean_loss.item():.6f}\t\t'
                           f'Val loss: {val_loss.item():.6f}\n'
                           f'*** Max memory reserved: {torch.cuda.max_memory_reserved() / 1024 / 1024}MB\n'
                           f'*** Reset optimizer. Learning rate adjusted: {current_lr} -> {new_lr}')

        # optimizer.zero_grad()
        bbar = tqdm(training_input, desc='Fitting batches', unit='batch', leave=False)
        for i, batch in enumerate(bbar):
            optimizer.zero_grad()
            data, target = batch[0].unsqueeze(2), batch[1]

            with torch.no_grad():
                nchunk = data.shape[1]//200
                # target = torch.cat((data.clone().detach(), data.clone().detach()), dim=2)
                data = torch.chunk(data.to(torch.float16) / 255, nchunk, dim=1)
                target = torch.chunk(target.to(torch.float16) / 255, nchunk, dim=1)

            for d, t in zip(data, target):

                # optimizer.zero_grad()

                with torch.autocast(dev):
                    pred = model(d)
                    loss = loss_function(pred, t)

                scaler.scale(loss).backward()
                # loss.backward()

                with torch.no_grad():
                    loss_list.append(loss.item())
                    bbar.set_postfix(Loss=f'{loss.item():.5f}')

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

        with torch.no_grad():
            mean_loss = np.mean(np.array(loss_list))
            val_loss = torch.tensor([0], device=dev)
            if n_val > 0:
                model.eval()
                val_batch = next(iter(validation_input))
                val_data, val_target = val_batch[0].unsqueeze(2).to(torch.float16) / 255, val_batch[1].to(torch.float16) / 255

                nchunk = val_data.shape[1] // 200
                # target = torch.cat((data.clone().detach(), data.clone().detach()), dim=2)
                val_data = torch.chunk(val_data, nchunk, dim=1)

                val_pred = []
                for d in val_data:
                    p = model(d)
                    val_pred.append(p)
                val_pred = torch.cat(val_pred, dim=1)

                val_loss = loss_function(val_pred, val_target)

            pbar.set_postfix(Loss=f'{mean_loss:.5f}',
                             Val_Loss=f'{val_loss.item():.5f}')

            checkpoint = {
                'state_dict': model.state_dict(),
            }
            torch.save(checkpoint, Path(__file__).parent / 'checkpoint.pt')

            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        if n_val > 0:
            val_batch = next(iter(validation_input))
        else:
            val_batch = next(iter(training_input))

        val_data, val_target = val_batch[0].unsqueeze(2).to(torch.float16) / 255, val_batch[1]
        val_target = val_target.cpu().detach().numpy()
        print(np.max(val_target))
        nchunk = val_data.shape[1] // 200
        # target = torch.cat((data.clone().detach(), data.clone().detach()), dim=2)
        val_data = torch.chunk(val_data, nchunk, dim=1)

        val_pred = []
        for d in val_data:
            p = model(d)
            p = torch.sigmoid(p)
            val_pred.append(p)
        val_pred = torch.cat(val_pred, dim=1)
        print(val_pred.shape, torch.max(val_pred), torch.max(val_pred[:, :, 0, ...]), torch.max(val_pred[:, :, 1, ...]))

        val_pred = (val_pred.cpu().detach().numpy() * 255)
        val_data = torch.cat(val_data, dim=1)
        val_data = val_data.cpu().detach().numpy() * 255

    print(f'\n*** Max memory reserved: {torch.cuda.max_memory_reserved() / 1024 / 1024}MB\n')

    return checkpoint, val_data, val_pred, val_target


def main():
    args = docopt(__doc__, version=f'LarvaPicker {__version__}: PostureTracker trainer')
    print(args, '\n')

    if torch.cuda.is_available():
        # Moving to GPU
        print('\n*** GPU available!\n')
        # dev = torch.device('cuda:0')
        dev = 'cuda'
    else:
        # dev = torch.device('cpu')
        dev = 'cpu'

    checkpoint_path = Path(args['--checkpoint_path']) \
        if args['--checkpoint_path'] else Path(__file__).parent / 'checkpoint.pt'
    override = args['--override'].lower() in ['true', '1', 't', 'y', 'yes']
    checkpoint = {
        'state_dict': None
    }
    if Path.is_file(checkpoint_path):
        print('\nPrevious checkpoint available.')
        if not (checkpoint_path.parent / 'backup').is_dir():
            Path.mkdir(checkpoint_path.parent / 'backup')
        now = datetime.datetime.now()
        now_ = now.strftime("%m_%d_%Y_%H_%M")
        shutil.copy(checkpoint_path, checkpoint_path.parent / 'backup' / f'checkpoint_{now_}.pt')
        if not override:
            checkpoint = torch.load(checkpoint_path)

    checkpoint, data, pred, target = train_model(
        index_path=Path(args['--index_path']) if args['--index_path'] else Path(__file__).parent / 'index.json',
        img_shape=(64, 64),
        n_channels_in=1,
        n_channels_out=2,
        init_nodes=int(args['--init_nodes']),
        dev=dev,
        n_epoch=int(args['--n_epoch']),
        lr_init=float(args['--lr_init']),
        batch_size=int(args['--batch_size']),
        validation_split=float(args['--val_split']),
        state_dict=checkpoint['state_dict'],
    )
    print(f'\nSaving checkpoint to: {checkpoint_path.as_posix()}')
    torch.save(checkpoint, checkpoint_path)

    if pred is not None:
        print('\nSaving prediction to video...')
        if not Path.is_dir(Path(__file__).parent / 'bin'):
            Path.mkdir(Path(__file__).parent / 'bin')
        in_rec = cv2.VideoWriter(
            str(Path(__file__).parent / 'bin' / f'{date} target.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            10, (64, 64), True
        )
        out_rec = cv2.VideoWriter(
            str(Path(__file__).parent / 'bin' / f'{date} pred.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            10, (64, 64), True
        )
        print(pred.shape, np.max(pred))
        for t in range(pred.shape[1]):
            # in_frame = data[0, t, 0, ...] + data[0, t, 1, ...]
            in_frame = np.dstack([target[0, t, 0, ...], target[0, t, 1, ...], np.zeros_like(data[0, t, 0, ...])])
            in_rec.write(np.uint8(np.clip(in_frame[:, :, ::-1], 0, 255)))
            out_frame = np.dstack([pred[0, t, 0, ...], pred[0, t, 1, ...], np.zeros_like(data[0, t, 0, ...])])
            out_rec.write(np.uint8(np.clip(out_frame[:, :, ::-1], 0, 255)))
        in_rec.release()
        out_rec.release()

    print('\n\n*** DONE!')


if __name__ == '__main__':
    main()
