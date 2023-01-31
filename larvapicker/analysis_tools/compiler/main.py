"""
Compiler: compile data from trackers into behavioral features of interest
You can also run Compiler as a module with the same arguments:
    compiler --dataset_path=<dataset_path> [options]

Usage:
    main.py -h | --help
    main.py -v | --version
    main.py (--dataset_path=<dataset_path> | --index_path=<index_path>) [options]

Options:
    -h --help                           show this message and exit.
    -v --version                        show version information and exit.
    --dataset_path=<dataset_path>       path to data directory to analyze.
    --index_path=<index_path>           path to index of all directories to analyze.
    --file_tag=<file_tag>               informational tag to add to end of saved files.
"""

import matplotlib.pyplot as plt
from docopt import docopt
from tqdm import tqdm

from .visualizers import plot_figure
from ..utils.io import *
from ...config.__version__ import __version__
from ...config.constants import px_to_mm


def compiler(
        dataset_path,
        ):

    tqdm.write(f'\nAnalyzing data at: {dataset_path}')

    with open(dataset_path.parent / 'log.json', 'r') as f:
        log = json.load(f)

    with h5py.File(dataset_path / 'coordinates.h5', 'r') as f:
        coordinates = np.zeros(f["data"].shape)
        f["data"].read_direct(coordinates)
    coordinates = np.array(coordinates)

    with h5py.File(dataset_path / "pt_results.h5", 'r') as f:
        pt_results = np.zeros(f["data"].shape)
        f["data"].read_direct(pt_results)
    pt_results = np.array(pt_results)

    with h5py.File(dataset_path / "st_results.h5", 'r') as f:
        st_results = np.zeros(f["data"].shape)
        f["data"].read_direct(st_results)
    st_results = np.array(st_results)

    lp_activity = np.empty((0, 2))
    trajectory = np.empty((0, 4))
    velocities = np.empty((0, 4))
    turns = np.empty((0, 3))
    tti = np.empty((0, 3))
    bba = np.empty((0, 2))

    t_pickup, t_dropoff = None, None
    status_prev = state_prev = t_prev = x_prev = y_prev = None
    t_turn_init, t_turn_end = None, None
    t_run_init, t_run_end = None, None
    x_run_init, x_run_end = None, None
    y_run_init, y_run_end = None, None
    for fn in tqdm(range(coordinates.shape[0]), desc='Analyzing behavior', unit='frames'):
        status = log[str(fn)]['status']
        t, x, y = coordinates[fn]

        tidx = np.where(pt_results[:, 0] == t)[0]
        if len(tidx) == 0:
            continue
        t, x, y, hx, hy, tx, ty = pt_results[tidx[0], :]
        state = st_results[tidx[0]]

        if status_prev is None:
            # first frame, skipping for analysis
            status_prev = status
            t_prev, x_prev, y_prev = t, x, y
            state_prev = state
            continue

        if status == 'PICKUP' and status_prev == 'INIT':
            # first pick up attempt, i.e. first contact with larva
            t_pickup = t
        elif (status in ['INIT', 'PICKUP']
              and status_prev not in ['PICKUP', 'DROPOFF']):
            # first pick up attempt, i.e. first contact with larva
            t_pickup = t
        elif (status not in ['INIT', 'PICKUP', 'DROPOFF']
                and status_prev in ['INIT', 'PICKUP', 'DROPOFF']):
            # last dropoff attempt, i.e. last contact with larva
            t_dropoff = t
            # checking that pickup/dropoff are from the same robot event
            if 0.0 < t_dropoff - t_pickup < 600.0:
                lp_activity = np.append(lp_activity, np.array([[t_pickup, t_dropoff]]), axis=0)

        if status in ['INIT', 'PICKUP', 'DROPOFF', 'DELIVER', 'FEED']:
            # robot activity, skipping for analysis
            status_prev = status
            t_prev, x_prev, y_prev = t, x, y
            state_prev = state
            continue

        if state == 0 and state_prev == 0:
            disp = np.array([x - x_prev, y - y_prev])
            norm = np.linalg.norm(disp)
            if norm < 8 and (t - t_prev) > 1e-8:
                # consecutive points must be within 8 pixels to avoid including jumps
                trajectory = np.append(trajectory, np.array([[t, state, x, y]]), axis=0)
                velocities = np.append(
                    velocities,
                    np.array([[
                        (t + t_prev) / 2,
                        (t - t_prev),
                        disp[0],
                        disp[1]
                    ]]),
                    axis=0
                )

                hv, tv = [hx-32, hy-32], [32-tx, 32-ty]
                bend_angle = np.arccos(np.dot(hv, tv) / (np.linalg.norm(hv)*np.linalg.norm(tv)))
                if not np.isnan(bend_angle):
                    bba = np.append(bba, np.array([[t, bend_angle*180/np.pi]]), axis=0)

            if t_run_init is None:
                t_run_init = t
                x_run_init = x
                y_run_init = y

        elif state == 1 and state_prev == 0:
            t_turn_init = t
            t_run_end = t
            x_run_end = x
            y_run_end = y
            if t_run_init is not None:
                tt_width = (t_run_end - t_run_init)
                temp_therm_grad = np.arctan2((y_run_end - y_run_init),
                                             (x_run_end - x_run_init))
                tti = np.append(tti, np.array([[tt_width/120, temp_therm_grad, t_run_end]]), axis=0)
            t_run_init, x_run_init, y_run_init = None, None, None

            if t_turn_end is None:
                pass
            elif t_turn_init - t_turn_end < 1.0:
                # a new turn cannot be initiated within 1 seconds of the last one
                t_turn_init = None

        elif state == 0 and state_prev == 1:
            if t_turn_init is not None:
                t_turn_end = t
                t_run_init = t
                x_run_init = x
                y_run_init = y

                vslice_i = np.where((pt_results[:, 0] > t_turn_init-5)*(pt_results[:, 0] < t_turn_init))[0]
                vslice_f = np.where((pt_results[:, 0] > t_turn_end)*(pt_results[:, 0] < t_turn_end+5))[0]
                turn_chi = 0
                turn_mag = 0
                if len(vslice_i) > 0 and len(vslice_f) > 0:
                    v_i = [pt_results[vslice_i[-1], 1] - pt_results[vslice_i[0], 1],
                           pt_results[vslice_i[-1], 2] - pt_results[vslice_i[0], 2], 0]
                    v_f = [pt_results[vslice_f[-1], 1] - pt_results[vslice_f[0], 1],
                           pt_results[vslice_f[-1], 2] - pt_results[vslice_f[0], 2], 0]
                    cross = np.cross(v_i, v_f)
                    dot = np.dot(v_i, v_f)

                    # turn handedness here
                    if cross[-1] > 0:
                        turn_chi = 1
                    elif cross[-1] < 0:
                        turn_chi = -1

                    # turn magnitude here
                    if np.linalg.norm(v_i)*np.linalg.norm(v_f) > 0:
                        turn_mag = np.arccos(dot / (np.linalg.norm(v_i)*np.linalg.norm(v_f)))
                        turn_mag = turn_mag * 180 / np.pi

                t_turn = (t_turn_end + t_turn_init) / 2
                turns = np.append(turns, np.array([[t_turn, turn_chi, turn_mag]]), axis=0)

        else:
            trajectory = np.append(trajectory, np.array([[t, state, x, y]]), axis=0)

        status_prev = status
        t_prev, x_prev, y_prev = t, x, y
        state_prev = state

    # smoothing velocities
    conv_width = int((velocities.shape[0] / velocities[-1, 0]) * 5)
    tqdm.write(f'*** Convolving speeds over {conv_width} frames')
    speeds = np.empty((velocities.shape[0]-(conv_width-1), 2))
    speeds[:, 0] = np.convolve(
        velocities[:, 0],
        np.ones(conv_width) / conv_width,
        mode='valid'
    )
    smooth_vels = np.empty((velocities.shape[0]-(conv_width-1), 2))
    for i in range(2):
        smooth_vels[:, i] = np.convolve(
            velocities[:, i+2],
            np.ones(conv_width),
            mode='valid'
        ) / np.convolve(
            velocities[:, 1],
            np.ones(conv_width),
            mode='valid'
        ) * px_to_mm
    speeds[:, 1] = np.linalg.norm(smooth_vels, axis=-1)

    smooth_traj = np.empty((trajectory.shape[0] - (conv_width-1), 3))
    smooth_traj[:, 0] = np.convolve(
        trajectory[:, 0],
        np.ones(conv_width) / conv_width,
        mode='valid'
    )
    for i in range(2):
        smooth_traj[:, 1+i] = np.convolve(
            trajectory[:, 2+i],
            np.ones(conv_width) / conv_width,
            mode='valid'
        ) * px_to_mm

    run_curvs = np.empty((0, 3))
    curv_bin = np.array([0, 20])
    while True:
        if smooth_traj[-1, 0] < curv_bin[0]:
            break

        traj_bin_slice = np.where((smooth_traj[:, 0] > curv_bin[0])*(smooth_traj[:, 0] < curv_bin[1]))[0]
        if len(traj_bin_slice) > 3:
            dx = np.gradient(smooth_traj[traj_bin_slice, 1])
            dy = np.gradient(smooth_traj[traj_bin_slice, 2])
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            curvature = np.abs(d2y*dx - d2x*dy) / (dx**2 + dy**2)**(3/2)

            curvature = curvature
            curvature = curvature[~np.isnan(curvature)]
            curvature = curvature[curvature < 100]
            if len(curvature) > 3:
                run_curvs = np.append(
                    run_curvs,
                    np.array([[np.mean(smooth_traj[traj_bin_slice, 0]),
                               np.mean(curvature), np.max(curvature)]]),
                    axis=0
                )

        curv_bin += (curv_bin[1] - curv_bin[0])

    turn_mags = np.empty((0, 3))
    turn_rates = np.empty((0, 2))
    turn_bin = np.array([0, 360])
    while True:
        if turns[-1, 0] < turn_bin[0]:
            break

        turn_bin_slice = np.where((turns[:, 0] > turn_bin[0])*(turns[:, 0] < turn_bin[1]))[0]
        turn_count = len(turn_bin_slice)
        if turn_count > 1:
            turn_t = (turns[turn_bin_slice[-1], 0] + turns[turn_bin_slice[0], 0])/2
            # turn_bin_width = (turns[turn_bin_slice[-1], 0] - turns[turn_bin_slice[0], 0])
            chi_t = np.mean(turns[turn_bin_slice, 1])
            mag_t = np.mean(turns[turn_bin_slice, 2])
        elif turn_count == 1:
            turn_t = turns[turn_bin_slice[0], 0]
            # turn_bin_width = (turn_bin[1] - turn_bin[0])
            chi_t = np.mean(turns[turn_bin_slice, 1])
            mag_t = np.mean(turns[turn_bin_slice, 2])
        else:
            turn_t = (turn_bin[0] + turn_bin[1])/2
            chi_t = 0
            mag_t = 0
        turn_bin_width = (turn_bin[1] - turn_bin[0])/120

        turn_mags = np.append(
            turn_mags,
            np.array([[turn_t, chi_t, mag_t]]),
            axis=0
        )
        turn_rates = np.append(
            turn_rates,
            np.array([[turn_t, turn_count / turn_bin_width]]),
            axis=0
        )
        turn_bin += (turn_bin[1] - turn_bin[0])
    plot_figure(
        turn_rates[:, 0] / 3600, turn_rates[:, 1], None,
        title='Turn rate vs Time', fmt='r.-',
        xlabel='Time (hr)', xlims=(0, np.ceil(turn_rates[-1, 0] / 3600)),
        ylabel='Turn rate (turns/min)', ylims=(0, 5.0), filename=None
    )
    plt.savefig(str(dataset_path / 'turnrate-vs-time.pdf'), bbox_inches='tight')
    plt.clf()
    plt.close()

    window_width = conv_width * 60
    lp_interaction = np.empty((0, 2, window_width))
    for (t_pickup, t_dropoff) in lp_activity:
        pu_idx = np.argmin(np.abs(trajectory[:, 0] - t_pickup))
        do_idx = np.argmin(np.abs(trajectory[:, 0] - t_dropoff))
        disp = np.sqrt((trajectory[do_idx, 2]-trajectory[pu_idx, 2])**2
                       + (trajectory[do_idx, 3]-trajectory[pu_idx, 3])**2)
        if disp < 128:
            continue

        pu_idx = np.argmin(np.abs(velocities[:, 0] - t_pickup))
        do_idx = np.argmin(np.abs(velocities[:, 0] - t_dropoff))
        if (pu_idx-window_width-conv_width < 0
                or do_idx+window_width+conv_width >= velocities.shape[0]):
            continue

        smooth_pu = np.empty((window_width, 2))
        smooth_do = np.empty((window_width, 2))
        for i in range(2):
            smooth_pu[:, i] = np.convolve(
                velocities[pu_idx-window_width-conv_width+1:pu_idx, i+2],
                np.ones(conv_width),
                mode='valid'
            ) / np.convolve(
                velocities[pu_idx-window_width-conv_width+1:pu_idx, 1],
                np.ones(conv_width),
                mode='valid'
            ) * px_to_mm

            smooth_do[:, i] = np.convolve(
                velocities[do_idx:do_idx+window_width+conv_width-1, i+2],
                np.ones(conv_width),
                mode='valid'
            ) / np.convolve(
                velocities[do_idx:do_idx+window_width+conv_width-1, 1],
                np.ones(conv_width),
                mode='valid'
            ) * px_to_mm

        lp_interaction = np.append(
            lp_interaction,
            np.append(
                np.linalg.norm(smooth_pu, axis=-1)[None, :],
                np.linalg.norm(smooth_do, axis=-1)[None, :],
                axis=0
            )[None, ...],
            axis=0
        )
    tqdm.write(f'*** {lp_interaction.shape[0]} robot interactions found')

    n_pts = 50
    interaction_bins = np.empty((n_pts*2, 0))
    arr_list = np.array_split(lp_interaction, n_pts, axis=-1)
    pre_interactions = np.zeros((n_pts, arr_list[0].shape[0]))
    post_interactions = np.zeros((n_pts, arr_list[0].shape[0]))
    for i, arr in enumerate(arr_list):
        pre_interactions[i] = np.mean(arr[:, 0, :], axis=-1)
        post_interactions[i] = np.mean(arr[:, 1, :], axis=-1)
    interaction_bins = np.append(
        interaction_bins,
        np.append(pre_interactions, post_interactions, axis=0),
        axis=-1
    )
    plot_figure(
        np.linspace(-5, 0, n_pts), np.mean(interaction_bins[:n_pts, :], axis=-1), None,
        title='Speed vs Time', fmt='b-',
        xlabel='Time (min)', xlims=(-5, 5),
        ylabel='Speed (mm/sec)', ylims=(0, 0.8), filename=None
    )
    plt.plot(np.linspace(0, 5, n_pts), np.mean(interaction_bins[n_pts:, :], axis=-1), 'r-')
    for i in range(interaction_bins.shape[1]):
        plt.plot(np.linspace(-5, 5, 2*n_pts), interaction_bins[:, i], alpha=0.2)
    plt.xticks(np.arange(-5, 6), fontsize='large')
    plt.axvline(0, color='black', alpha=0.3, linestyle='--')
    plt.savefig(str(dataset_path / 'lp-interactions.pdf'), bbox_inches='tight')
    plt.clf()
    plt.close()

    therm_grad = np.array([1.0, 0])     # 0.03499
    v_filter = np.where(speeds[:, 1] > 1e-8)[0]
    tni = np.append(
        speeds[v_filter, 0, None],
        np.divide(
            np.sum(smooth_vels[v_filter, :] * therm_grad[None, :], axis=-1),
            speeds[v_filter, 1]
        )[:, None],
        axis=-1
    )
    mean_t = []
    mean_tni = []
    std_tni = []
    t_bin = np.array([0, 240])
    while t_bin[0] < np.max(tni[:, 0]):
        therm_bin_slice = np.where((tni[:, 0] > t_bin[0])
                                   * (tni[:, 0] < t_bin[1]))[0]
        if len(therm_bin_slice) > 0:
            mean_t.append(np.mean(tni[therm_bin_slice, 0]))
            mean_tni.append(np.mean(tni[therm_bin_slice, 1]))
            std_tni.append(np.std(tni[therm_bin_slice, 1]))
        t_bin += (t_bin[1] - t_bin[0])

    return speeds, turn_rates, run_curvs, turn_mags, bba, lp_interaction, tni, tti


def main():
    args = docopt(__doc__, version=f'LarvaPicker {__version__}: Compiler')
    print(args, '\n')

    if args['--index_path'] is not None:
        index_path = Path(args['--index_path'])
        with open(index_path) as json_file:
            index = json.load(json_file)
        index = [Path(i) for i in index.values()]
    else:
        index = [Path(args['--dataset_path'])]

    if not Path.is_dir(Path(__file__).parent / 'bin'):
        Path.mkdir(Path(__file__).parent / 'bin')

    if args['--file_tag'] is not None:
        tag = f'-{args["--file_tag"]}'
    else:
        tag = ''

    (all_speeds, all_turns, all_run_curv, all_turn_mag, all_bba,
     all_interactions, all_therms, all_tti) = [], [], [], [], [], [], [], []

    for path in tqdm(index, desc='Compiling data', unit='dataset'):
        n = 0
        while True:
            p = path / str(n)
            if not p.is_dir():
                break

            # speeds, turn_rates, lp_interaction, tni, tti = compiler(dataset_path=p)
            (speeds, turn_rates, run_curv, turn_mag, bba,
             lp_interaction, tni, tti) = compiler(dataset_path=p)

            all_speeds.append(speeds)
            all_turns.append(turn_rates)
            all_run_curv.append(run_curv)
            all_turn_mag.append(turn_mag)
            all_bba.append(bba)
            all_interactions.append(lp_interaction)
            all_therms.append(tni)
            all_tti.append(tti)

            n += 1

    print('\nAll datasets analyzed! Compiling population statistics...')

    t_bin = np.array([0, 120])
    t_max = np.max([np.max(all_speeds[i][:, 0]) for i in range(len(all_speeds))])

    mean_times = []
    dwell_threshold = 0.01
    mean_speeds = []
    std_speeds = []
    mean_turn_rates = []
    std_turn_rates = []
    mean_dweller_frac = []
    std_dweller_frac = []
    mean_tnis = []
    std_tnis = []

    mean_run_curvs = []
    std_run_curvs = []
    mean_turn_chis = []
    std_turn_chis = []
    mean_turn_mags = []
    std_turn_mags = []
    mean_body_angles = []
    std_body_angles = []

    corr_matrix = []
    tni_matrix = []

    while True:
        if t_max < t_bin[1]:
            break

        mean_times.append((t_bin[0] + t_bin[1])/2/3600)

        corr_matrix_t = [[], []]
        tni_matrix_t = []

        # compiling crawling speeds and dwelling probability,
        # i.e. fraction of data points per animal spent dwelling
        mean_speed = []
        weight_speed = []
        std_speed = []
        dweller_frac = []
        for i, speeds in enumerate(all_speeds):
            speed_bin_slice = np.where((speeds[:, 0] > t_bin[0])
                                       * (speeds[:, 0] < t_bin[1]))[0]
            std_bin = []
            dweller_count = 0
            for s in speeds[speed_bin_slice, 1]:
                if s > dwell_threshold:
                    mean_speed.append(s)
                    weight_speed.append(1/len(speed_bin_slice))
                    std_bin.append(s)
                else:
                    dweller_count += 1

            if len(std_bin) > 0:
                std_speed.append(np.mean(std_bin))
                corr_matrix_t[0].append(np.mean(std_bin))
            else:
                std_speed.append(0)
                corr_matrix_t[0].append(0)
            if len(speed_bin_slice) > 0:
                dweller_frac.append(dweller_count / len(speed_bin_slice))
        # mean_speeds.append(np.average(mean_speed, weights=weight_speed))
        # std_speeds.append(np.std(std_speed))
        mean_speeds.append(np.mean(std_speed))
        std_speeds.append(0)
        mean_dweller_frac.append(np.average(dweller_frac))
        std_dweller_frac.append(np.std(dweller_frac))

        # compiling turn rates
        mean_turn_rate = []
        std_turn_rate = []
        for i, turn_rates in enumerate(all_turns):
            turn_bin_slice = np.where((turn_rates[:, 0] > t_bin[0])
                                      * (turn_rates[:, 0] < t_bin[1]))[0]
            std_bin = []
            for r in turn_rates[turn_bin_slice, 1]:
                mean_turn_rate.append(r)
                std_bin.append(r)
            if len(std_bin) > 0:
                std_turn_rate.append(np.mean(std_bin))
                corr_matrix_t[1].append(np.mean(std_bin))
            else:
                std_turn_rate.append(0)
                corr_matrix_t[1].append(0)
        if len(mean_turn_rate) > 0:
            mean_turn_rates.append(np.mean(mean_turn_rate))
            std_turn_rates.append(np.std(std_turn_rate))
        elif len(mean_turn_rates) > 0:
            mean_turn_rates.append(mean_turn_rates[-1])
            std_turn_rates.append(std_turn_rates[-1])
        else:
            mean_turn_rates.append(0)
            std_turn_rates.append(0)

        # compiling run segment curvatures
        mean_run_curv = []
        std_run_curv = []
        for i, curvs in enumerate(all_run_curv):
            curv_bin_slice = np.where((curvs[:, 0] > t_bin[0])
                                      * (curvs[:, 0] < t_bin[1]))[0]
            std_bin = []
            for r in curvs[curv_bin_slice, 1]:
                if not np.isnan(r):
                    mean_run_curv.append(r)
                    std_bin.append(r)
            std_run_curv.append(np.mean(std_bin))
        if len(mean_run_curv) > 0:
            mean_run_curvs.append(np.mean(mean_run_curv))
            std_run_curvs.append(np.std(std_run_curv))
        elif len(mean_run_curvs) > 0:
            mean_run_curvs.append(mean_run_curvs[-1])
            std_run_curvs.append(std_run_curvs[-1])
        else:
            mean_run_curvs.append(0)
            std_run_curvs.append(0)

        # compiling turn magnitudes
        mean_turn_chi = []
        std_turn_chi = []
        mean_turn_mag = []
        std_turn_mag = []
        for i, mags in enumerate(all_turn_mag):
            mag_bin_slice = np.where((mags[:, 0] > t_bin[0])
                                      * (mags[:, 0] < t_bin[1]))[0]
            std_chi_bin = []
            std_mag_bin = []
            for c, m in zip(mags[mag_bin_slice, 1], mags[mag_bin_slice, 2]):
                if not np.isnan(c) and not np.isnan(m):
                    if c != 0 and m != 0:
                        mean_turn_chi.append(c)
                        std_chi_bin.append(c)
                        mean_turn_mag.append(m)
                        std_mag_bin.append(m)
            if len(std_chi_bin) > 0:
                std_turn_chi.append(np.mean(std_chi_bin))
                std_turn_mag.append(np.mean(std_chi_bin))
            else:
                std_turn_chi.append(0)
                std_turn_mag.append(0)
        if len(mean_turn_chi) > 0:
            mean_turn_chis.append(np.mean(mean_turn_chi))
            std_turn_chis.append(np.std(std_turn_chi))
            mean_turn_mags.append(np.mean(mean_turn_mag))
            std_turn_mags.append(np.std(std_turn_mag))
        elif len(mean_turn_chis) > 0:
            mean_turn_chis.append(mean_turn_chis[-1])
            std_turn_chis.append(std_turn_chis[-1])
            mean_turn_mags.append(mean_turn_mags[-1])
            std_turn_mags.append(std_turn_mags[-1])
        else:
            mean_turn_chis.append(0)
            std_turn_chis.append(0)
            mean_turn_mags.append(0)
            std_turn_mags.append(0)

        # compiling body bend angle
        mean_bba = []
        std_bba = []
        for i, angles in enumerate(all_bba):
            bba_bin_slice = np.where((angles[:, 0] > t_bin[0])
                                     * (angles[:, 0] < t_bin[1]))[0]
            std_bin = []
            for a in angles[bba_bin_slice, 1]:
                if not np.isnan(a):
                    mean_bba.append(a)
                    std_bin.append(a)
            std_bba.append(np.mean(std_bin))
        if len(mean_bba) > 0:
            mean_body_angles.append(np.mean(mean_bba))
            std_body_angles.append(np.std(std_bba))
        elif len(mean_body_angles) > 0:
            mean_body_angles.append(mean_body_angles[-1])
            std_body_angles.append(std_body_angles[-1])
        else:
            mean_body_angles.append(0)
            std_body_angles.append(0)


        # compiling thermotaxis navigation index
        mean_tni = []
        weight_tni = []
        std_tni = []
        for therms in all_therms:
            therm_bin_slice = np.where((therms[:, 0] > t_bin[0])
                                       * (therms[:, 0] < t_bin[1]))[0]
            std_bin = []
            for i in therms[therm_bin_slice, 1]:
                mean_tni.append(i)
                weight_tni.append(1 / len(therm_bin_slice))
                std_bin.append(i)
            if len(std_bin) > 0:
                std_tni.append(np.mean(std_bin))
                tni_matrix_t.append(np.mean(std_bin))
            else:
                std_tni.append(0)
                tni_matrix_t.append(-10)
        mean_tnis.append(np.average(mean_tni))
        std_tnis.append(np.std(std_tni))

        t_bin += (t_bin[1] - t_bin[0])
        corr_matrix.append(corr_matrix_t)
        tni_matrix.append(tni_matrix_t)

    corr_matrix = np.transpose(corr_matrix, (2, 1, 0))
    corrcoef = []
    for matrix in corr_matrix:
        corrcoef.append(np.corrcoef(np.array(matrix))[0, 1])
    print(f'\nIndividual corrcoef: {corrcoef}'
          f'\n\nAveraged corrcoef: {np.mean(corrcoef):.4f}+/-{np.std(corrcoef)/2:.4f}')
    print(f'Corrcoef of averages: '
          f'{np.corrcoef(np.array([mean_speeds, mean_turn_rates]))[0, 1]:.4f}')

    # compiling speeds around robot interaction
    n_data = 200
    compiled_interactions = np.empty((2*n_data, 0))
    for interaction in all_interactions:
        arr_list = np.array_split(interaction, n_data, axis=-1)
        pre_interactions = np.zeros((n_data, arr_list[0].shape[0]))
        post_interactions = np.zeros((n_data, arr_list[0].shape[0]))
        for i, arr in enumerate(arr_list):
            pre_interactions[i] = np.mean(arr[:, 0, :], axis=-1)
            post_interactions[i] = np.mean(arr[:, 1, :], axis=-1)

        d = (post_interactions[0, :] - np.mean(pre_interactions, axis=0))
        compiled_interactions = np.append(
            compiled_interactions,
            np.append(pre_interactions[:, :], post_interactions[:, :], axis=0),
            axis=-1
        )
    mean_interactions = np.mean(compiled_interactions, axis=-1)
    std_interactions = np.std(compiled_interactions, axis=-1)
    print(f'Analyzed a total of {compiled_interactions.shape[-1]} robot interactions.')

    print('\nPlotting and saving results...')

    plot_figure(
        np.array(mean_times), np.array(mean_speeds), np.array(std_speeds),
        title='Speed vs Time', fmt='b-',
        xlabel='Time (hr)', xlims=(0, 36),
        ylabel='Speed (mm/sec)', ylims=(0, 0.8),
        filename=f'speed-vs-time{tag}'
    )
    plot_figure(
        np.array(mean_times), np.array(mean_turn_rates), np.array(std_turn_rates),
        title='Turn rate vs Time', fmt='r-',
        xlabel='Time (hr)', xlims=(0, 36),
        ylabel='Turn rate (turns/min)', ylims=(0, 3.0),
        filename=f'turnrate-vs-time{tag}'
    )
    plot_figure(
        np.array(mean_times), np.array(mean_run_curvs), np.array(std_run_curvs),
        title='Run curvature vs Time', fmt='r-',
        xlabel='Time (hr)', xlims=(0, 36),
        ylabel='Curvature', ylims=(0, 10.0),
        filename=f'curvature-vs-time{tag}'
    )
    plot_figure(
        np.array(mean_times), np.array(mean_turn_chis), np.array(std_turn_chis),
        title='Turn handedness vs Time', fmt='r-',
        xlabel='Time (hr)', xlims=(0, 36),
        ylabel='Turn handedness', ylims=(-1.0, 1.0),
        filename=f'turnchi-vs-time{tag}'
    )
    plot_figure(
        np.array(mean_times), np.array(mean_turn_mags), np.array(std_turn_mags),
        title='Turn magnitude vs Time', fmt='r-',
        xlabel='Time (hr)', xlims=(0, 36),
        ylabel='Turn magnitude (deg)', ylims=(0, 180.0),
        filename=f'turnmag-vs-time{tag}'
    )
    plot_figure(
        np.array(mean_times), np.array(mean_body_angles), np.array(std_body_angles),
        title='Body bend angle vs Time', fmt='r-',
        xlabel='Time (hr)', xlims=(0, 36),
        ylabel='Body bend angle (deg)', ylims=(0, 180.0),
        filename=f'bba-vs-time{tag}'
    )
    plot_figure(
        np.array(mean_times), np.array(mean_dweller_frac), np.array(std_dweller_frac),
        title='Dwelling fraction vs Time', fmt='g-',
        xlabel='Time (hr)', xlims=(0, 6),
        ylabel='Dweller fraction', ylims=(0, 0.5),
        filename=f'dwell-vs-time{tag}'
    )
    plot_figure(
        np.array(mean_times), np.array(mean_tnis), np.array(std_tnis),
        title=f'Thermotaxis navigation index vs Time '
              f'(avg: {np.mean(mean_tnis):.4f}'
              f'+/-{np.mean(std_tnis)/2/np.sqrt(len(std_tnis)):.4f})',
        fmt='r-', xlabel='Time (hr)', xlims=(0, 6),
        ylabel='TNI (v * dT)', ylims=(-1.0, 1.0),
        filename=f'tni-vs-time{tag}', close=False
    )
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    plt.savefig(str(Path(__file__).parent / 'bin' / f'tni-vs-time{tag}.pdf'), bbox_inches='tight')
    plt.clf()
    plt.close()

    plot_figure(
        np.linspace(-5, 0, n_data), mean_interactions[:n_data], std_interactions[:n_data],
        title='Speed vs Time', fmt='b-',
        xlabel='Time (min)', xlims=(-5, 5),
        ylabel='Speed (mm/sec)', ylims=(0, 0.8), filename=None
    )
    plt.plot(np.linspace(0, 5, n_data), mean_interactions[n_data:], 'r-')
    plt.fill_between(
        np.linspace(0, 5, n_data),
        mean_interactions[n_data:]-std_interactions[n_data:]/2,
        mean_interactions[n_data:]+std_interactions[n_data:]/2,
        color='r', alpha=0.2
    )
    plt.xticks(np.arange(-5, 6), fontsize='large')
    plt.axvline(0, color='black', alpha=0.3, linestyle='--')
    plt.savefig(str(Path(__file__).parent / 'bin' / f'lp_interactions{tag}.pdf'), bbox_inches='tight')
    plt.clf()
    plt.close()
    np.save(str(Path(__file__).parent / 'bin' / f'lp_interactions{tag}.npy'),
                np.stack((mean_interactions, std_interactions), axis=0), allow_pickle=True)

    print('\n\n*** DONE!')


if __name__ == '__main__':
    main()
