import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

Limits = namedtuple('Limits', 'low high')
Labels = namedtuple('Labels', 'x y')
Channel = namedtuple('Channel', 'name color')


class Screen:
    """Single screen for displaying telemetry."""
    def __init__(self, ax, title, xlimits=Limits(0,1), ylimits=Limits(0,1), channel_names=[Channel('y','k')],
        labels=Labels('x','y'), n_update_decay=5):
        self.ax = ax
        self.channel_names = channel_names
        self.n_update_decay = n_update_decay

        self.ax.set_title(title)
        self.ax.grid(axis='y')

        self.ax.set_xlim(xlimits)
        self.ax.set_xlabel(labels.x)

        self.ax.set_ylim(ylimits)
        self.ax.set_ylabel(labels.y)

    def on_episode_start(self):
        """Make old lines to disappear. Call on start of every episode."""
        n_channels = len(self.channel_names)
        assert (len(self.ax.lines) % n_channels == 0), \
            "Lines on axes not matching with N*n_channels, where N is integer"

        # Remove oldest update. Pop all channels from single update.
        if len(self.ax.lines) > self.n_update_decay*n_channels:
            for i in range(n_channels):
                self.ax.lines.pop(0)

        # Sliding transparency from newest update to oldest
        n_lines = len(self.ax.lines)
        n_updates = n_lines  / n_channels
        alphas = np.repeat(np.linspace(0, 1, n_updates), n_channels)
        for i in range(n_lines):
            self.ax.lines[i].set_alpha(alphas[i])

    def update(self, t, channels):
        """Update monitor with new data. Note that you need to draw and flus canvas after this."""
        # Fix 1-dimensonal arrays
        if len(channels.shape) == 1:
            channels = channels.reshape(1,-1)
        assert (channels.shape[0] == len(self.channel_names)), "Given channels must match with channel names list"
        assert (channels.shape[1] == len(t)), "Length of time vector must match with length of channel data"

        for i in range(channels.shape[0]):
            self.ax.plot(t, channels[i], label=self.channel_names[i].name,
                alpha=1.0, color=self.channel_names[i].color)


class Panel:
    """Panel including many telemetry screens that share time axis."""
    def __init__(self, t_limits, n_update_decay=5):
        self.screens = []

        time_label = 'time (s)'
        z_limits = Limits(0, 20)
        xy_limits = Limits(-4, 4)
        cost_limits = Limits(-10, 10)
        nrad_limits = Limits(-1, 1)
        rot_limits = Limits(-10, 910)
        xyz_labels = Labels(time_label, 'm')
        nrad_labels = Labels(time_label, 'nrad')
        cost_labels = Labels(time_label, '')
        rot_labels = Labels(time_label, 'rev/s')

        # Create subplot grid. Share x axis for each column.
        self.fig, self.ax = plt.subplots(3, 3, sharex='col', figsize=(8,8))

        # z-coordinate
        self.screens.append(Screen(self.ax[0,0], 'z-coord', t_limits, z_limits,
            [Channel('z', 'xkcd:sky blue')], xyz_labels, n_update_decay))
        # xy-coordinates
        ch = [Channel('x', 'r'), Channel('y', 'g')]
        self.screens.append(Screen(self.ax[1,0], 'xy-coord', t_limits, xy_limits, ch,
            xyz_labels, n_update_decay))
        # phi and theta
        self.screens.append(Screen(self.ax[0,1], 'phi', t_limits, nrad_limits,
            [Channel('', 'c')], nrad_labels, n_update_decay))
        self.screens.append(Screen(self.ax[1,1], 'theta', t_limits, nrad_limits,
            [Channel('', 'm')], nrad_labels, n_update_decay))
        # reward
        self.screens.append(Screen(self.ax[0,2], 'Reward', t_limits, cost_limits,
            [Channel('', 'xkcd:deep blue')], cost_labels, n_update_decay))
        self.screens.append(Screen(self.ax[1,2], 'Lateral cost', t_limits, cost_limits,
            [Channel('', 'xkcd:sea green')], cost_labels, n_update_decay))
        self.screens.append(Screen(self.ax[2,2], 'Attitude cost', t_limits, cost_limits,
            [Channel('', 'xkcd:coral')], cost_labels, n_update_decay))
        # rotation speeds
        ax = plt.subplot(3,3,(7,8))
        ch = [Channel('prop1','xkcd:puce'), Channel('prop2','xkcd:cerulean'),
            Channel('prop3','xkcd:goldenrod'), Channel('prop4','xkcd:fuchsia')]
        self.screens.append(Screen(ax, 'Rotation speeds', t_limits, rot_limits,
            ch, rot_labels, n_update_decay))

        # Set styles, font and scaling
        plt.rc('grid', linestyle='dashed', color='black')
        plt.rc('font', size=10)
        plt.subplots_adjust(wspace=0.5, hspace=0.3)

    def on_episode_start(self):
        for s in self.screens:
            s.on_episode_start()

    def update(self, t, channels):
        # z, xy
        self.screens[0].update(t, channels[2])
        self.screens[1].update(t, channels[0:2])
        # phi, theta
        self.screens[2].update(t, channels[3])
        self.screens[3].update(t, channels[4])
        # reward, lat, att
        self.screens[4].update(t, channels[6])
        self.screens[5].update(t, channels[7])
        self.screens[6].update(t, channels[8])
        # rotation speeds
        self.screens[7].update(t, channels[9:])
        # Draw and flush
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
