

# TODO: Should this class be removed and have the Runner build the operators and run them?
class OperationFactory:
    """
    This is a factory for building the different type of operations that can be performed by the pipeline
    """
    def __init__(self, available_operator_map, application, dm):
        self._available_operator_map = available_operator_map
        self._application = application
        self._dm = dm

    def _get_operation_class(self, operation_name):
        if operation_name not in self._available_operator_map:
            raise ValueError("Unknown Operation type: %s" % str(operation_name))

        return self._available_operator_map[operation_name]

    def build(self, operation_name):
        op = self._get_operation_class(operation_name)
        return op(self._application, self._dm)
