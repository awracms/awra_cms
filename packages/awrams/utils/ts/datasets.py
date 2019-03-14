from awrams.utils.metatypes import ObjectDict ,ObjectContainer

class AWRAVariableInstance(object):
    def __init__(self,source,name,units):
        self.source = source
        self.name = name
        self.units = units
        self.meta = ObjectDict()

    def get_data(self,period,extent): #pylint: disable=unused-argument
        raise Exception("Not implemented")

    def __repr__(self):
        return self.name

class VariableGroup(ObjectContainer):
    '''
    Iterable container for variables, with metadata
    '''

    def __init__(self,source,variables=None):
        if variables is None:
            variables = []

        self.source = source
        ObjectContainer.__init__(self)
        for v in variables:
            self[v.name] = v
