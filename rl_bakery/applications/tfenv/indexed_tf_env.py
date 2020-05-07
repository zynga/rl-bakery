class IndexedTFEnv:
    def __init__(self, tf_env, env_id):
        self.tf_env = tf_env
        self.env_id = env_id
        self.tf_env.reset()
        self.ts_id = 0

    def step(self, action_step):
        """
        Apply an action to current environment.
        Params:
            action: PolicyStep
        Returns:
            time_step (dict). this will contain the users state and reward before that action was take
        """
        tf_env_ts = self.tf_env.current_time_step()

        time_step = {
            "env_id": self.env_id,
            "ts_id": self.ts_id,
            "reward": tf_env_ts.reward.numpy()[0].item(),
            "step_type": tf_env_ts.step_type.numpy()[0].item(),
            "discount": tf_env_ts.discount.numpy()[0].item(),
            "action": action_step.action.numpy()[0].tolist()
        }

        if action_step.info:
            time_step["policy_info"] = {}
            for k in action_step.info:
                meta_tensor = action_step.info[k]
                time_step["policy_info"][k] = meta_tensor.numpy()[0].tolist()

        # add observation to time_step. Each feature is a key in time_step
        ob_list = tf_env_ts.observation.numpy()[0]
        for (i, ob) in enumerate(ob_list):
            time_step["ob_{}".format(i)] = ob.item()

        action = action_step.action.numpy()
        self.tf_env.step(action)
        self.ts_id += 1
        return time_step
