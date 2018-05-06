import sys
import numpy as np
from collections import deque
from telemetry import Panel, Limits
import matplotlib.pyplot as plt


class Trainer:
    """Trainer."""
    def __init__(self, agent, show_graph=False, show_stats=False):
        self.agent = agent
        self.show_graph = show_graph
        self.show_stats = show_stats
        self.scores = []
        self.panel = None

    def train(self, num_episodes=20, display_freq=2, n_update_decay=5):
        """Train agent."""
        self.scores = deque(maxlen=num_episodes)
        best_score = -np.inf
        sim = self.agent.task.sim
        # Telemetry data
        telem = deque(maxlen=int(sim.runtime/sim.dt))
        # Create telemetry panel
        if self.show_graph:
            self.panel = Panel(Limits(0, sim.runtime), n_update_decay)

        for i_episode in range(1, num_episodes+1):
            # Housekeeping for starting fresh episode
            state = self.agent.reset_episode()
            if self.show_graph:
                telem.clear()
                self.panel.on_episode_start()

            while True:
                # Take action with OU noise
                action = self.agent.act(state)
                # Interact with environment
                next_state, reward, done = self.agent.task.step(action)
                # Update telemetry data. Use normalized Eulers.
                if (i_episode % display_freq == 0) and self.show_graph:
                    telem.append(np.hstack([sim.time, sim.pose[:3],
                        self.agent.task.normalize_angles(sim.pose[3:]),
                        reward, action]))
                # Run batch learning step
                self.agent.step(action, reward[0], next_state, done)
                # Roll over next state
                state = next_state

                if done:
                    # Use mean reward per episode as score
                    score = self.agent.total_reward / self.agent.n_steps
                    self.scores.append(score)
                    if score > best_score:
                        best_score = score

                    if (i_episode % display_freq == 0):
                        if self.show_graph:
                            # Variables on rows
                            rows = np.vstack(telem).T
                            # Update panel
                            self.panel.update(rows[0], rows[1:])

                        if self.show_stats:
                            # Print episode statistics to console
                            print("Episode: {:4d}, end time: {:2.1f}, score: {:7.3f} (best: {:7.3f})".format(
                                i_episode, sim.time, score, best_score))
                            sys.stdout.flush()
                    break
        # End of learning, save weights and scores
        self.save_results()

    def save_results(self):
        """Save learning results."""
        self.agent.save()
        filename = "scores.npz"
        np.savez(filename, self.scores)
        print("Saved learning scores to: ",filename)

    def plot_scores(self, mav_filter=True, mav_window=10):
        """
        Plot scores from latest training.

        :param mav_filter: use moving average filtering
        :param mav_window: moving average window length (default: 10)
        """
        return plot_scores(self.scores, mav_filter, mav_window, self.agent.name)


class Evaluator:
    """Evaluate trained agent in simulator."""
    def __init__(self, agent):
        self.agent = agent
        self._labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
            'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
            'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
        self.reset()

    def load_weights(self):
        """Load saved weights to given agent."""
        try:
            self.agent.load()
        except Exception as e:
            print("Cannot find weight files: ", e)

    def reset(self):
        """Clear simulation results."""
        self._saved_results = {x: [] for x in self._labels}

    def run(self):
        """Run agent in simulator with given task."""
        state = self.agent.reset_episode()
        sim = self.agent.task.sim

        while True:
            # choose action without OU noise
            rotor_speeds = self.agent.act(state, add_noise=False)

            # record current state
            for ii in range(len(self._labels)):
                to_write = [sim.time]+list(sim.pose)+list(sim.v)+list(sim.angular_v)+list(rotor_speeds)
                self._saved_results[self._labels[ii]].append(to_write[ii])

            # interact with environment
            next_state, _, done = self.agent.task.step(rotor_speeds)
            # roll over state
            state = next_state
            if done:
                print("Simulation end time: {:7.2f} s".format(sim.time))
                break

    def plot_dynamics(self, results=None, alpha=0.9):
        """Plot quadcopter dynamics from saved simulation results."""
        titles = ['Position', 'Euler angles', 'Lateral velocities', 'Angular velocities',
            'Rotor speeds']
        ylabels = ['m', 'rad', 'm/s', 'rad/s', 'rev/s']
        # Use saved results if no argument given
        if results is None:
            results=self._saved_results
        t = results[self._labels[0]]

        fig = plt.figure(figsize=(12,15))
        fig.suptitle("Quadcopter dynamics")
        # Create 3x2 subplot grid for dynamics plots. Iterate all except revs.
        p_i = 0
        for i in range(1,10+1,3):
            ax = fig.add_subplot(3,2,p_i+1)
            ax.plot(t, results[ self._labels[i]], label=self._labels[i], alpha=alpha)
            ax.plot(t, results[ self._labels[i+1]], label=self._labels[i+1], alpha=alpha)
            ax.plot(t, results[ self._labels[i+2]], label=self._labels[i+2], alpha=alpha)
            ax.grid(axis='y')
            plt.title(titles[p_i])
            plt.xlabel('time'+' / s')
            plt.ylabel(ylabels[p_i])
            plt.legend()
            p_i += 1

        # Plot motor revs. Leaves one subplot still empty.
        ax = fig.add_subplot(3,2,5)
        for i in range(13,16+1):
            ax.plot(t, results[ self._labels[i]], label=self._labels[i], alpha=alpha)
        ax.grid(axis='y')
        plt.title(titles[p_i])
        plt.xlabel('time')
        plt.ylabel(ylabels[p_i])
        plt.legend()


def plot_scores(scores, mav_filter=True, mav_window=10, name=None):
    """Helper method for plotting scores with moving average filtering."""
    def moving_average(a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    fig = plt.figure(figsize=(10,5))
    if mav_filter:
        fig.suptitle("Scores (MAV window size: {})".format(mav_window))
        y = moving_average(scores, mav_window)
        x = np.linspace(mav_window, len(scores), len(y))
    else:
        fig.suptitle("Scores")
        y = scores
        x = [x for x in range(1,len(scores)+1)]

    ax = fig.add_subplot(111)
    ax.plot(x, y, label=name)
    ax.grid(axis='y')
    if name is not None:
        ax.legend()
    plt.rc('grid', linestyle='dashed', color='black')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Score')
