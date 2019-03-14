from awrams.utils.nodegraph import graph,nodes
from awrams.utils import mapping_types as mt
from awrams.utils import extents
from awrams.utils.metatypes import ObjectDict as odict

class OnDemandSimulator:
    def __init__(self,model,imapping,omapping=None,extent=None):

        if extent is None:
            extent = extents.get_default_extent()

        imapping = graph.get_input_tree(model.get_input_keys(),imapping)
        #+++
        #Document the use of this manually, don't just change the graph behind the scenes...
        #imapping = graph.map_rescaling_nodes(imapping,extent)

        self.input_runner = graph.ExecutionGraph(imapping)
        self.model_runner = model.get_runner(self.input_runner.get_dataspecs(True))

        self.outputs = None
        if omapping is not None:
            self.outputs = graph.OutputGraph(omapping)

        self._dspecs = self.input_runner.get_dataspecs(True)

    def run(self,period,extent,return_inputs=False,expanded=True):
        coords = mt.gen_coordset(period,extent)
        if self.outputs:
            ### initialise output files if necessary
            self.outputs.initialise(period,extent)

        iresults = self.input_runner.get_data_flat(coords,extent.mask)
        mresults = self.model_runner.run_from_mapping(iresults,coords.shape[0],extent.cell_count,recycle_states=False)

        if self.outputs is not None:
            self.outputs.set_data(coords,mresults,extent.mask)

        if expanded:
            mresults = nodes.expand_dict(mresults,extent.mask)
            if return_inputs:
                ires_spatial = dict([(k,iresults[k]) for k,v in self._dspecs.items() if 'cell' in v.dims])
                ires_other = dict([(k,iresults[k]) for k,v in self._dspecs.items() if 'cell' not in v.dims])
                iresults = nodes.expand_dict(ires_spatial,extent.mask)
                for k,v in ires_other.items():
                    if k not in self.input_runner.const_inputs:
                        iresults[k] = v

        if return_inputs:
            return mresults,iresults
        else:
            return mresults

    def run_prepack(self,iresults,period,extent):
        '''
        run with pre-packaged inputs for calibration
        :param cid:
        :param period:
        :param extent:
        :return:
        '''
        coords = mt.gen_coordset(period,extent)
        if self.outputs:
            ### initialise output files if necessary
            self.outputs.initialise(period,extent)

        mresults = self.model_runner.run_from_mapping(iresults,coords.shape[0],extent.cell_count)

        if self.outputs is not None:
            self.outputs.set_data(coords,mresults,extent.mask)

        return mresults
