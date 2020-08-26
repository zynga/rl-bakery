
from hyperopt import hp

sim_config = {
    # Cartpole has no configurations
}

#
# A list of param combinations for GridSearch.
#
hp_config = [
    {
        "learning_rate": 0.01,
        "mini_batch_size": 64,
        "fc_layer_params": (100,)
    },
    {
        "learning_rate": 0.02,
        "mini_batch_size": 32,
        "fc_layer_params": (50,)
    },
    {
        "learning_rate": 0.05,
        "mini_batch_size": 32,
        "fc_layer_params": (100,)
    }
]