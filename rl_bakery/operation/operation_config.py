class OperationConfig:
    # TODO: param is not needed
    def __init__(self, op_type, run_id):
        self._op_name = op_type
        self._run_id = run_id

    @property
    def op_name(self):
        return self._op_name

    @property
    def run_id(self):
        return self._run_id

    def __repr__(self):
        return "OperationConfig(%s, TS: %s)" % (self.op_name, self.run_id)

    def __eq__(self, other):
        return (self.op_name, self.run_id) == (other.op_name, other.run_id)
