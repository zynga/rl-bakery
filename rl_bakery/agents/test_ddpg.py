
from rl_bakery.agents.ddpg import DDPGAgent, DDPGConfig
from rl_bakery.applications import agent_application
from rl_bakery.agents.abstract import Optimizer
from tf_agents import specs
import unittest
import mock


class TestDDPGAgent(unittest.TestCase):

    @mock.patch('rl_bakery.agents.ddpg.DdpgAgent', autospec=True)
    @mock.patch('rl_bakery.agents.ddpg.ActorNetwork', autospec=True)
    @mock.patch('rl_bakery.agents.ddpg.CriticNetwork', autospec=True)
    def test_default_config(self, mock_critic_network, mock_actor_network, mock_agent):
        params = ["agent.actor_fc_layer_params=[100,10]", "agent.observation_fc_layer_params=[1,2,3]",
                  "agent.action_fc_layer_params=[1,2,3,4]", "agent.joint_fc_layer_params=[5]"]

        obs_spec = "obs_spec"
        dataspec = agent_application.DataSpec(
            observation_spec=specs.ArraySpec([1,2,3], int),
            action_spec=specs.ArraySpec([1], float)
        )
        conf = agent_application.make_config(DDPGConfig(), params)

        agent_trainer = DDPGAgent(dataspec, conf)
        agent = agent_trainer.init_agent()

        mock_critic_network.assert_called_once_with((dataspec.observation_spec, dataspec.action_spec),
                                                    observation_fc_layer_params=[1,2,3],
                                                    action_fc_layer_params=[1,2,3,4],
                                                    joint_fc_layer_params=[5])

        mock_actor_network.assert_called_once_with(dataspec.observation_spec, dataspec.action_spec,
                                                   fc_layer_params=[100,10])
        mock_agent.assert_called_once_with(
            time_step_spec=mock.ANY, # TODO
            action_spec=dataspec.action_spec,
            train_step_counter=mock.ANY, # TODO
            actor_network=mock_actor_network.return_value,
            critic_network=mock_critic_network.return_value,
            actor_optimizer=mock.ANY, #TODO
            critic_optimizer=mock.ANY, #TODO
            td_errors_loss_fn=None,
            target_actor_network=None,
            target_critic_network=None,
        )
        self.assertEqual(agent, mock_agent.return_value)

    @mock.patch('rl_bakery.agents.ddpg.DdpgAgent', autospec=True)
    @mock.patch('rl_bakery.agents.ddpg.ActorNetwork', autospec=True)
    @mock.patch('rl_bakery.agents.ddpg.CriticNetwork', autospec=True)
    def test_default_config(self, mock_critic_network, mock_actor_network, mock_agent):
        params = ["agent.actor_fc_layer_params=[100,10]", "agent.observation_fc_layer_params=[1,2,3]",
                  "agent.action_fc_layer_params=[1,2,3,4]", "agent.joint_fc_layer_params=[5]",
                  "agent.ou_stddev=0.1",
                  "agent.ou_damping=0.2",
                  "agent.target_update_tau=0.3",
                  "agent.target_update_period=1",
                  "agent.dqda_clipping=1.1",
                  "agent.reward_scale_factor=1.2",
                  "agent.gradient_clipping=1.3",
                  "agent.debug_summaries=True",
                  "agent.summarize_grads_and_vars=True",
                  "agent.name=Patrick"]

        obs_spec = "obs_spec"
        dataspec = agent_application.DataSpec(
            observation_spec=specs.ArraySpec([1,2,3], int),
            action_spec=specs.ArraySpec([1], float)
        )
        conf = agent_application.make_config(DDPGConfig(), params)

        agent_trainer = DDPGAgent(dataspec, conf)
        agent = agent_trainer.init_agent()

        mock_critic_network.assert_called_once_with((dataspec.observation_spec, dataspec.action_spec),
                                                    observation_fc_layer_params=[1,2,3],
                                                    action_fc_layer_params=[1,2,3,4],
                                                    joint_fc_layer_params=[5])

        mock_actor_network.assert_called_once_with(dataspec.observation_spec, dataspec.action_spec,
                                                   fc_layer_params=[100,10])
        mock_agent.assert_called_once_with(
            time_step_spec=mock.ANY, # TODO
            action_spec=dataspec.action_spec,
            train_step_counter=mock.ANY, # TODO
            actor_network=mock_actor_network.return_value,
            critic_network=mock_critic_network.return_value,
            actor_optimizer=mock.ANY, #TODO
            critic_optimizer=mock.ANY, #TODO
            td_errors_loss_fn=None,
            target_actor_network=None,
            target_critic_network=None,
            ou_stddev=0.1,
            ou_damping=0.2,
            target_update_tau=0.3,
            target_update_period=1,
            dqda_clipping=1.1,
            reward_scale_factor=1.2,
            gradient_clipping=1.3,
            debug_summaries=True,
            summarize_grads_and_vars=True,
            name="Patrick"
        )
        self.assertEqual(agent, mock_agent.return_value)

