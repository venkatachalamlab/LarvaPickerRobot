"""
StateTracker trainer: trains StateTracker's core neural network with flagged data from StateFlagger.

Usage:
    train.py -h | --help
    train.py -v | --version
    train.py [options]

Options:
    -h --help                           show this message and exit.
    -v --version                        show version information and exit.
    --index_path=<index_path>           path to index of data directories with flagged data.
    --checkpoint_path=<checkpoint_path> path to training checkpoint file.
    --init_nodes=<init_nodes>           number of kernels for the first layer. [default: 8]
    --dropout_rate=<dropout_rate>       fraction of intermediate data to drop out during training. [default: 0]
    --n_epoch=<n_epoch>                 number of training epochs. [default: 100]
    --lr_init=<lr_init>                 initial learning rate. [default: 0.1]
    --batch_size=<batch_size>           batch size. [default: 1]
    --val_split=<val_split>             fraction of data to use for validation. [default: 0]
    --override=<override>               override existing training checkpoint. [default: False]
"""


import datetime
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from docopt import docopt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .model import StateTracker
from ..utils.io import *
from ..utils.streamers import MultiDataStreamer
from ...config.__version__ import __version__
from ...config.constants import date


def train_model(
        index_path=Path('./index.json'),
        n_channels_in=7,
        n_channels_out=1,
        init_nodes=8,
        dropout_rate=0.,
        dev='cpu',
        n_epoch=100,
        lr_init=0.1,
        batch_size=1,
        validation_split=0.,
        state_dict=None
        ):

    print('\n\nCompiling model...')
    model = StateTracker(
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        n_chunks=1,
        init_nodes=init_nodes,
        dropout_rate=dropout_rate
    ).to(dev)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3], device=dev))
    optimizer = optim.Adadelta(model.parameters(), lr=lr_init, eps=1e-6)
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

        model.train()
        with torch.no_grad():
            loss_list = []

            if epoch > 0 and epoch % 4 == 0:
                tqdm.write(f'Epoch: {epoch}\t\t'
                           f'Loss: {mean_loss.item():.4f}\t\t'
                           f'Val loss: {val_loss.item():.4f}')
            # if epoch > 0 and epoch % 16 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                optimizer = optim.Adadelta(model.parameters(), lr=lr_init, eps=1e-6)
                scaler = torch.cuda.amp.GradScaler()
                new_lr = optimizer.param_groups[0]['lr']
                tqdm.write(f'*** Max memory reserved: {torch.cuda.max_memory_reserved()/1024/1024}MB')
                tqdm.write(f'*** Reset optimizer. Learning rate adjusted: {current_lr} -> {new_lr}')

        bbar = tqdm(training_input, desc='Fitting batches', unit='batch', leave=False)
        for i, batch in enumerate(bbar):
            optimizer.zero_grad()

            data, target = batch[0].to(torch.float16), batch[1].to(torch.float16)

            with torch.no_grad():
                n_chunks = 1
                data = data.chunk(n_chunks, dim=1)
                target = target.chunk(n_chunks, dim=1)

            for d, t in zip(data, target):
                # print(d.shape)

                # optimizer.zero_grad()

                # pred = model(d)
                # loss = loss_function(pred, t)
                # loss.backward()
                with torch.autocast(dev):
                    pred = model(d)
                    loss = loss_function(pred, t)

                with torch.no_grad():
                    loss_list.append(loss.clone().detach())
                    bbar.set_postfix(Loss=f'{loss.item():.4f}')

                scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            mean_loss = torch.mean(torch.tensor(loss_list, device=dev))
            val_loss = torch.tensor([0], device=dev)
            if n_val > 0:
                model.eval()
                val_batch = next(iter(validation_input))
                val_data, val_target = val_batch[0].to(dev).to(torch.float32), val_batch[1].to(dev).to(torch.float32)
                val_pred = model(val_data)
                val_loss = loss_function(val_pred, val_target)
            pbar.set_postfix(Loss=f'{mean_loss.item():.4f}',
                             Val_Loss=f'{val_loss.item():.4f}')

            checkpoint = {
                'state_dict': model.state_dict(),
            }
            torch.save(checkpoint, Path(__file__).parent / 'checkpoint.pt')

    model.eval()
    with torch.no_grad():
        if n_val > 0:
            val_batch = next(iter(validation_input))
        else:
            val_batch = next(iter(training_input))
        val_data, val_target = val_batch[0].to(dev), val_batch[1].to(dev)
        val_pred = model(val_data.to(torch.float32))
        val_pred = torch.sigmoid(val_pred)
        val_data = val_data.cpu().detach().numpy()[0, ...]
        val_pred = val_pred.cpu().detach().numpy()[0, ...]
        val_target = val_target.cpu().detach().numpy()[0, ...]

    return checkpoint, val_data, val_pred, val_target


def main():
    args = docopt(__doc__, version=f'LarvaPicker {__version__}: StateTracker trainer')
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
        'state_dict': None,
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
        n_channels_in=7,
        n_channels_out=1,
        init_nodes=int(args['--init_nodes']),
        dropout_rate=float(args['--dropout_rate']),
        dev=dev,
        n_epoch=int(args['--n_epoch']),
        lr_init=float(args['--lr_init']),
        batch_size=int(args['--batch_size']),
        validation_split=float(args['--val_split']),
        state_dict=checkpoint['state_dict']
    )
    print(f'\nSaving checkpoint to: {checkpoint_path.as_posix()}')
    torch.save(checkpoint, checkpoint_path)

    print('\nVisualizing prediction...')
    if not Path.is_dir(Path(__file__).parent / 'bin'):
        Path.mkdir(Path(__file__).parent / 'bin')

    if data is not None and pred is not None and target is not None:
        # pred = np.rint(pred).astype(int).flatten()
        print(np.max(pred), np.mean(pred), np.sum(pred))
        pred = np.where(pred.flatten() > 0.5, 1, 0).astype(int)
        target = np.rint(target).astype(int).flatten()

        plt.figure(0)
        plt.title('state_flagger results')
        plt.scatter(data[(target == 0), 1], data[(target == 0), 2],
                    color='g', marker='o', s=2, alpha=0.33)
        plt.scatter(data[(target == 1), 1], data[(target == 1), 2],
                    color='darkorange', marker='o', s=1, alpha=0.66)
        plt.savefig((Path(__file__).parent / 'bin' / f'{date} target.png').as_posix(), dpi=300)

        plt.figure(1)
        plt.title('state_tracker results')
        plt.scatter(data[(pred == 0), 1], data[(pred == 0), 2],
                    color='g', marker='o', s=2, alpha=0.33)
        plt.scatter(data[(pred == 1), 1], data[(pred == 1), 2],
                    color='darkorange', marker='o', s=1, alpha=0.66)
        plt.savefig((Path(__file__).parent / 'bin' / f'{date} pred.png').as_posix(), dpi=300)
        plt.close()

    print('\n\n*** DONE!')


if __name__ == '__main__':
    main()
