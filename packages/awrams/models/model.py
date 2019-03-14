class Model:

    def __init__(self):
        pass

    def get_runner(self,dataspecs,shared=False):
        """
        Return a ModelRunner for this model
        
        Args:
            dataspecs (dict): Dataspecs (as returned from ExecutionGraph)
            shared (bool): Is this runner being used in a shared memory context?

        Returns:
            ModelRunner
        """
        raise NotImplementedError

    def init_shared(self,dataspecs):
        """
        Perform any shared initialisation required before multiple instances of get_runner are called

        Args:
            dataspecs (dict): Dataspecs (as returned from ExecutionGraph)
        """
        pass

    def get_input_keys(self):
        """
        Return the list of keys required as inputs

        Returns:
            list

        """
        raise NotImplementedError

    def get_state_keys(self):
        """
        Return the list of keys representing model states

        Returns:
            list
        """
        return []

    def get_output_variables(self):
        """
        Return the list of output variable keys for this model

        Returns:
            list
        """
        raise NotImplementedError

    def get_default_mapping(self):
        """
        Return the default input mapping for this model
        This is a dict of key:GraphNode mappings

        Return:
            mapping (dict)
        """
        raise NotImplementedError

    def get_output_mapping(self):
        """
        Return a mapping of key/model_output nodes for this model
        """
        from awrams.utils.nodegraph.nodes import model_output

        outmap = {}
        for k in self.get_output_variables():
            outmap[k] = model_output(k)

        return outmap
        
class ModelRunner:

    def run_from_mapping(self,mapping,timesteps,cells):
        """
        Run the model over the supplied key/ndarray mapping of inputs,
        for 

        Args:
            mapping (dict): key/ndarray pairs of input data
            timesteps (int): number of timesteps to run for
            cells (int): number of cells

        Returns:
            outputs (dict): dictionary of key/ndarray pairs for model outputs,
                            optionally with subdict 'final_states' for stateful models
        """
        raise NotImplementedError