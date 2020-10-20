import unittest
import mock
from rl_bakery.applications import agent_application
from rl_bakery.agents.dqn import DDQNAgent, DQNAgent, QConfig
from rl_bakery.agents.abstract import Optimizer
from tf_agents import specs


class TestDQN(unittest.TestCase):

    def test_qconfig_values(self):
        params = ["agent.optimizer.learning_rate=0.01",
                  "agent.fc_layer_params=[100, 150, 90]",
                  ]

        conf = agent_application.make_config(QConfig(), params)
        self.assertEqual(conf.agent.optimizer.learning_rate, 0.01)
        # default is Adam
        self.assertEqual(conf.agent.optimizer.optimizer, Optimizer.Adam)
        self.assertEqual(conf.agent.fc_layer_params, [100, 150, 90])

    @mock.patch('rl_bakery.agents.dqn.DqnAgent', autospec=True)
    @mock.patch('rl_bakery.agents.abstract.QNetwork', autospec=True)
    def test_standard_config_dqn(self, mock_qnetwork, mock_agent):
        params = ["agent.optimizer.learning_rate=0.01",
                  "policy.epsilon_greedy=0.01",
                  "trajectory.n_step=1",
                  "agent.boltzmann_temperature=200",
                  "agent.emit_log_probability=True",
                  "agent.target_update_tau=1.0",
                  "agent.target_update_period=2",
                  "agent.gamma=1.1",
                  "agent.reward_scale_factor=1.2",
                  "agent.gradient_clipping=1.5",
                  "agent.debug_summaries=True",
                  "agent.summarize_grads_and_vars=False",
                  "agent.name=Patrick",
                  "agent.fc_layer_params=[100, 150, 90]",
                  ]

        dataspec = agent_application.DataSpec(
            observation_spec=specs.ArraySpec([1,2,3], int),
            action_spec=specs.ArraySpec([1], float)
        )
        conf = agent_application.make_config(QConfig(), params)

        agent_trainer = DQNAgent(dataspec, conf)
        agent = agent_trainer.init_agent()

        mock_qnetwork.assert_called_once_with(dataspec.observation_spec, dataspec.action_spec, fc_layer_params=[100,150,90])
        mock_agent.assert_called_once_with(
            time_step_spec=mock.ANY, # TODO
            action_spec=dataspec.action_spec,
            q_network=mock_qnetwork.return_value,
            train_step_counter=mock.ANY, # TODO
            optimizer=mock.ANY, #TODO
            epsilon_greedy=0.01,
            n_step_update=1,
            boltzmann_temperature=200,
            emit_log_probability=True,
            target_update_tau=1.0,
            target_update_period=2,
            gamma=1.1,
            reward_scale_factor=1.2,
            gradient_clipping=1.5,
            debug_summaries=True,
            summarize_grads_and_vars=False,
            name="Patrick",
        )
        self.assertEqual(agent, mock_agent.return_value)


    @mock.patch('rl_bakery.agents.dqn.DqnAgent', autospec=True)
    @mock.patch('rl_bakery.agents.abstract.QNetwork', autospec=True)
    def test_ignore_missing_config_dqn(self, mock_qnetwork, mock_agent):
        params = ["agent.fc_layer_params=[100, 150, 90]"]

        dataspec = agent_application.DataSpec(
            observation_spec=specs.ArraySpec([1,2,3], int),
            action_spec=specs.ArraySpec([1], float)
        )
        conf = agent_application.make_config(QConfig(), params)

        agent_trainer = DQNAgent(dataspec, conf)
        agent = agent_trainer.init_agent()

        mock_qnetwork.assert_called_once_with(dataspec.observation_spec, dataspec.action_spec, fc_layer_params=[100,150,90])
        mock_agent.assert_called_once_with(
            time_step_spec=mock.ANY, # TODO
            action_spec=dataspec.action_spec,
            q_network=mock_qnetwork.return_value,
            train_step_counter=mock.ANY, # TODO
            optimizer=mock.ANY, #TODO
            epsilon_greedy=conf.policy.epsilon_greedy,
            n_step_update=conf.trajectory.n_step
        )
        self.assertEqual(agent, mock_agent.return_value)

    @mock.patch('rl_bakery.agents.dqn.DdqnAgent', autospec=True)
    @mock.patch('rl_bakery.agents.abstract.QNetwork', autospec=True)
    def test_standard_config_ddqn(self, mock_qnetwork, mock_agent):
        params = ["agent.optimizer.learning_rate=0.01",
                  "policy.epsilon_greedy=0.01",
                  "trajectory.n_step=1",
                  "agent.boltzmann_temperature=200",
                  "agent.emit_log_probability=True",
                  "agent.target_update_tau=1.0",
                  "agent.target_update_period=2",
                  "agent.gamma=1.1",
                  "agent.reward_scale_factor=1.2",
                  "agent.gradient_clipping=1.5",
                  "agent.debug_summaries=True",
                  "agent.summarize_grads_and_vars=False",
                  "agent.name=Patrick",
                  "agent.fc_layer_params=[100, 150, 90]",
                  ]

        dataspec = agent_application.DataSpec(
            observation_spec=specs.ArraySpec([1,2,3], int),
            action_spec=specs.ArraySpec([1], float)
        )
        conf = agent_application.make_config(QConfig(), params)

        agent_trainer = DDQNAgent(dataspec, conf)
        agent = agent_trainer.init_agent()

        mock_qnetwork.assert_called_once_with(dataspec.observation_spec, dataspec.action_spec, fc_layer_params=[100,150,90])
        mock_agent.assert_called_once_with(
            time_step_spec=mock.ANY, # TODO
            action_spec=dataspec.action_spec,
            q_network=mock_qnetwork.return_value,
            train_step_counter=mock.ANY, # TODO
            optimizer=mock.ANY, #TODO
            epsilon_greedy=0.01,
            n_step_update=1,
            boltzmann_temperature=200,
            emit_log_probability=True,
            target_update_tau=1.0,
            target_update_period=2,
            gamma=1.1,
            reward_scale_factor=1.2,
            gradient_clipping=1.5,
            debug_summaries=True,
            summarize_grads_and_vars=False,
            name="Patrick",
        )
        self.assertEqual(agent, mock_agent.return_value)


    @mock.patch('rl_bakery.agents.dqn.DdqnAgent', autospec=True)
    @mock.patch('rl_bakery.agents.abstract.QNetwork', autospec=True)
    def test_ignore_missing_config_ddqn(self, mock_qnetwork, mock_agent):
        params = ["agent.fc_layer_params=[100, 150, 90]"]

        dataspec = agent_application.DataSpec(
            observation_spec=specs.ArraySpec([1,2,3], int),
            action_spec=specs.ArraySpec([1], float)
        )
        conf = agent_application.make_config(QConfig(), params)

        agent_trainer = DDQNAgent(dataspec, conf)
        agent = agent_trainer.init_agent()

        mock_qnetwork.assert_called_once_with(dataspec.observation_spec, dataspec.action_spec, fc_layer_params=[100,150,90])
        mock_agent.assert_called_once_with(
            time_step_spec=mock.ANY, # TODO
            action_spec=dataspec.action_spec,
            q_network=mock_qnetwork.return_value,
            train_step_counter=mock.ANY, # TODO
            optimizer=mock.ANY, #TODO
            epsilon_greedy=conf.policy.epsilon_greedy,
            n_step_update=conf.trajectory.n_step
        )
        self.assertEqual(agent, mock_agent.return_value)

