import matplotlib.pyplot as plt

from ..utils.io import *


def plot_figure(
    xdata, ydata, edata=None,
    title=None, fmt='b.-',
    xlabel=None, xlims=(0, 10),
    ylabel=None, ylims=(0, 1),
    filename=None, close=True):

    fig = plt.figure()
    if title is not None:
        plt.title(title)

    plt.plot(xdata, ydata, fmt)
    if edata is not None:
        plt.fill_between(
            xdata, ydata - edata/2, ydata + edata/2,
            color=fmt[0], alpha=0.2
        )

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize='x-large')
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize='x-large')
    plt.xticks(np.arange(np.floor(xdata[0]), np.ceil(xdata[-1])+1), fontsize='large')
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])

    if filename is not None:
        p = str(Path(__file__).parent / 'bin' / filename)
        np.save(p + '.npy', np.stack((xdata, ydata, edata), axis=0), allow_pickle=True)
        if close:
            plt.tight_layout()
            plt.savefig(p + '.pdf', bbox_inches='tight')
            plt.clf()
            plt.close()

    return fig
