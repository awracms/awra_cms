class Waitable:
    def __init__(self,persist=False):
        self.persist = persist

    def done(self):
        raise NotImplementedError

    def on_complete(self):
        raise NotImplementedError

class ConditionWaitable(Waitable):
    def __init__(self,condition,action):
        self.condition = condition
        self.action = action

    def done(self):
        if condition:
            self.on_complete()
        return condition

    def on_complete(self):
        self.action(*self.action_args)

class MPIWaitable(Waitable):
    def __init__(self,mpi_env,req,action=None,action_args=None):
        self.MPI = mpi_env
        self.req = req
        if action is None:
            action = self._no_action
        self.action = action
        if action_args is None:
            action_args = []
        self.action_args = action_args

    def _no_action(self,*args):
        pass

    def test_req(self):
        return self.req.Test()

    def done(self):
        done = self.test_req()
        if done:
            self.action(*self.action_args)
        return done

    def __repr__(self):
        return str(self.action_args)

class MPIMultiWaitable(MPIWaitable):

    def test_req(self):
        return self.MPI.Request.Testall(self.req)