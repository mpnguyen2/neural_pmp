import numpy as np
import math
from os import path
from classical_controls import ContinuousEnv
from gym import utils
from gym.utils import seeding
from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box

#### Ant Environment from Mujoco Gym ####
class Ant(ContinuousEnv):
    # Metadata is directly from Mujoco library
    metadata ={
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 20,
    }
    def _init__(self, q_dim=1, u_dim=1, control_coef=0.5):
        super().__init__(q_dim, u_dim, control_coef)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(111,), dtype=np.float64)
        MuJocoPyEnv.__init__(self, "ant.xml", 5, observation_space=observation_space)
        utils.EzPickle.__init__(self)

    # Dynamics f    
    def f(self, q, u):
        # qs
        self.xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(u, self.frame_skip)
        self.xposafter = self.get_body_com("torso")[0]

        self.ctrl_cost = 0.5 * np.square(u).sum()
        self.contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        # obs: 
        #   * qpos (12): XYZ and orientation (XYZW) of torso (4) and joint angles (8)
        #   * qvel (14): torso velocities (3:xyz), angular velocitiries (3:xyz) joint velocities (8)
        #   * cfrc_ext (14*6): external forces (force x,y,z and torque x,y,z) applied to each of the links at the center of mass.
        #       * the ground link, the torso link, and 12 links for all legs (3 links for each leg).
        obs = np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat, np.clip(self.sim.data.cfrc_ext, -1, 1).flat,])
        return obs

    #  Partial derivative of dynamics f wrt control u. Assuming linear control
    def f_u(self, q):
        
        pass #TODO

    # Lagrangian or running cost L
    def L(self, q, u):
        self.forward_reward = (self.xposafter - self.xposbefore) / self.dt
        return -(self.forward_reward - self.ctrl_cost - self.contact_cost + 1)

    # Terminal cost
    # TODO
    def g(self, q):
        zposition = self.state_vector()[2]
        terminal = (not (0.2 <= zposition <= 1.0)) and np.isfinite(self.state_vector())