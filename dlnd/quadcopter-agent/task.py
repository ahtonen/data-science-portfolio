import sys, copy
import numpy as np
from physics_sim import PhysicsSim


class Takeoff:
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, debug=False):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        self.sim = PhysicsSim(init_pose, None, None, 5.0)
        self.action_repeat = 1

        self.state_size = self.action_repeat * 5
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.debug = debug

    def normalize_angles(self, euler_angles):
        """Normalize Euler angles from 0,2*pi to -1,1 range."""
        angles = copy.copy(euler_angles)

        for ii in range(len(angles)):
            angles[ii] = angles[ii] if (angles[ii] < np.pi) else (angles[ii]-2*np.pi)
            angles[ii] /= np.pi
        return angles

    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward."""

        # Normalize roll and pitch to [-1,1]
        norm_eulers = self.normalize_angles(self.sim.pose[3:5+1])
        if self.debug:
            # Print latest position and normalized Euler angles
            print("(x',y',z')=({:6.2f},{:6.2f},{:6.2f}), (phi,theta,ksi)=({:6.2f},{:6.2f},{:6.2f})".format(
                *self.sim.v[:3], *norm_eulers))
            sys.stdout.flush()

        # Cost of excessive roll or pitch. Normalized to [-5,0].
        attitude_cost = -(5/2)*sum(abs(norm_eulers[:2]))
        # Cost of lateral movement ignored for now.
        # normalize to -5,0
        lateral_cost = -sum(abs(self.sim.v[0:2]))
        # Reward for z-speed up minus cost of rolling or pitching
        reward = self.sim.v[2] + attitude_cost
        #reward = self.sim.v[2] + 1./np.std(rotor_speeds)+ lateral_cost + attitude_cost

        # Reward for this step + debugging info
        return reward, lateral_cost, attitude_cost

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = np.zeros(3)
        pose_all = []

        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += np.array(self.get_reward(rotor_speeds))
            #pose_all.append(self.sim.pose)
            pose_all.append(np.hstack([self.sim.v, self.sim.pose[3:5]]))

        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = np.concatenate([self.sim.pose] * self.action_repeat)
        state = np.concatenate([np.hstack([self.sim.v, self.sim.pose[3:5]])] * self.action_repeat)
        return state


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
