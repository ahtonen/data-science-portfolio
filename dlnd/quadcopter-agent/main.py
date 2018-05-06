import numpy as np
from agents.agent import DDPG
from task import Takeoff
from trainer import Trainer
import matplotlib.pyplot as plt
import sys, getopt

#
# Launcher for running learning directly from command prompt
#
def print_usage():
    print("main.py -n <no. iterations> -d <display interval> -a <start altitude>")

def main(argv):
    # Training parameters
    n_iterations = 10
    display_freq = 2
    # Starting position
    init_pose = np.zeros(6)
    init_pose[2] = 10.0
    # Parse arguments
    try:
        opts, args = getopt.getopt(argv,"hn:d:a:")
    except getopt.GetoptError:
        sys.exit(2)
        print_usage()

    for opt, arg in opts:
        if opt == "-h":
            print_usage()
            sys.exit()
        elif opt == "-n":
            n_iterations = int(arg)
        elif opt == "-d":
            display_freq = int(arg)
        elif opt == "-a":
            init_pose[2] = float(arg)

    task = Takeoff(init_pose)
    agent = DDPG(task)
    trainer = Trainer(agent, show_graph=True, show_stats=True)

    print("\n\nStarting DDPG training for {:4d} iterations, z(t_0)={:4.1f}m".format(
        n_iterations, init_pose[2]))

    plt.ion()
    trainer.train(n_iterations, display_freq=display_freq, n_update_decay=3)
    plt.ioff()

if __name__ == "__main__":
   main(sys.argv[1:])
   # Close plot window only after keypress
   input("Press key to terminate...")
