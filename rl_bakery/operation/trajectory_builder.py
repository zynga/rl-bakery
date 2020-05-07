import copy
import numpy as np

# TODO: move this to a central location so that it can be reused by the build_timestep as well
TERMINAL_STATE = np.asarray(2, dtype=np.int32)


class TrajectoryStructure(object):
    """
    This structure facilitates building trajectories with shape [T,<datapoint shape>] given a list timesteps.
    It stores all trajectory data points in a queue. It also has a function to export a trajectory as a dictionary
    """

    def __init__(self, obs_cols=None, n_step=None):
        self.step_type = []
        self.observation = []
        self.action = []
        self.next_step_type = []
        self.reward = []
        self.discount = []
        self.policy_info = {}

        self.obs_cols = obs_cols
        self.n_step = n_step

    def add_timestep(self, ts):
        """
        add a timestep to the trajectory. if the trajectory is full, the oldest data points
        will be removed.

        Params:
            ts: new timestep
        """
        # only add these values if it is not the first timestep
        if len(self.observation) > 0:
            self.reward.append(ts["reward"])
            self.next_step_type.append(ts["step_type"])
            self.discount.append(ts["discount"])

        self.step_type.append(ts["step_type"])
        self.observation.append([ts[feature_name] for feature_name in self.obs_cols])
        self.action.append(ts["action"])

        if "policy_info" in ts:
            # Note: Policy info is a dictionary of arrays such as {"prob": [0.1, 0.9]}
            if not self.policy_info:
                self.policy_info = {k: [] for k in ts["policy_info"]}

            for k in ts["policy_info"]:
                self.policy_info[k].append(ts["policy_info"][k])

        # remove first item if list size is larger than n + 1
        if len(self.observation) > self.n_step + 1:
            self.reward = self.reward[1:]
            self.next_step_type = self.next_step_type[1:]
            self.discount = self.discount[1:]
            self.step_type = self.step_type[1:]
            self.observation = self.observation[1:]
            self.action = self.action[1:]
            if len(self.policy_info) > 0:
                for k in self.policy_info:
                    self.policy_info[k] = self.policy_info[k][1:]

    def is_complete(self):
        """
        A trajectory is complete when it n transitions were taken

        Return:
             True if yes, False otherwise
        """
        return len(self.observation) == self.n_step + 1

    def get_trajectory(self):
        """
        return the trajectory as a dict
        """
        # add fake values for last rewards, next_step_type and discount
        next_step_type = copy.deepcopy(self.next_step_type) + [self.next_step_type[-1]]
        reward = copy.deepcopy(self.reward) + [self.reward[-1]]
        discount = copy.deepcopy(self.discount) + [self.discount[-1]]

        result = {
            "step_type": copy.deepcopy(self.step_type),
            "observation": copy.deepcopy(self.observation),
            "action": copy.deepcopy(self.action),
            "policy_info": copy.deepcopy(self.policy_info),
            "reward": reward,
            "discount": discount,
            "next_step_type": next_step_type,
        }

        return result

    def is_terminated(self):
        return True if self.step_type and self.step_type[-1] == TERMINAL_STATE else False


class TrajectoryBuilder(object):
    """
    This class builds all possible trajectories given a set of timesteps.
    A trajectory structure is returned with all data points stored as a
    numpy array with dimension [B, T, <data point shape>].

    The columns representing the observation and environment/timestep identifiers
    identifiers are needed.
    """

    traj = None

    def __init__(self, obs_cols, env_id_cols, ts_id_col, n_step):
        self.env_id_cols = env_id_cols
        self.ts_id_col = ts_id_col
        self.obs_cols = obs_cols
        self.n_step = n_step
        self.traj = TrajectoryStructure()

    def run(self, timestep_list):

        trajectory_list = []

        current_traj = None
        current_traj_env_id = None

        for ts in timestep_list:
            ts_env_id = self._extract_env_id(ts)

            if ts_env_id != current_traj_env_id or current_traj.is_terminated():
                current_traj = TrajectoryStructure(self.obs_cols, self.n_step)
                current_traj_env_id = ts_env_id

            current_traj.add_timestep(ts)

            if current_traj.is_complete():
                traj = current_traj.get_trajectory()
                trajectory_list.append(traj)

        if len(trajectory_list) == 0:
            return None

        trajectory_dict = {k: [traj[k] for traj in trajectory_list] for k in trajectory_list[0] if k != "policy_info"}

        sample_ts = timestep_list[0]
        if "policy_info" in sample_ts:
            # Note: Policy info is a dictionary of arrays such as {"prob": [0.1, 0.9]}
            trajectory_dict["policy_info"] = {k: [traj["policy_info"][k] for traj in trajectory_list]
                                              for k in sample_ts["policy_info"]}
        else:
            trajectory_dict["policy_info"] = ()

        return trajectory_dict

    def _extract_env_id(self, ts):
        """
        Extract the values used to identify the environment id

        Param:
            ts: a Timestep

        Return:
            list of values representing the id of the environment
        """
        env_id_cols = self.env_id_cols
        return [ts[col] for col in env_id_cols]
