from rl_bakery.data_manager.data_manager import DataManager, DATANAME, InMemoryStorage, TFAgentStorage


def build_inmemory_data_manager(application):
    """
    Setup a DataManager to store intermediate date in memory. This data will be lost once the program closes.
    The trained agent is stored on disk at 'application.config.project.dm_storage_path/agent'

    :arg application: An AgentApplication

    Return: DataManager instance
    """
    storage_path = application.config.project.dm_storage_path + '/agent'

    dm = DataManager()
    dm.add_data(DATANAME.MODEL, TFAgentStorage(application.agent, storage_path, "agent.model"))
    dm.add_data(DATANAME.RUN_CONTEXT, InMemoryStorage())
    dm.add_data(DATANAME.TIMESTEP, InMemoryStorage())
    return dm
