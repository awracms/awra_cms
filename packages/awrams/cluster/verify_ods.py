"""
Simple script to verify that running an OnDemandSimulator
will produce the same results as our cluster sim
"""


from awrams.utils import datetools as dt
from awrams.utils import extents
import numpy as np
from awrams.utils.nodegraph import nodes, graph
from awrams.simulation.ondemand import OnDemandSimulator

full_extent = extents.get_default_extent()

period = dt.dates('dec 2010 - jan 2011')
extent = full_extent.ioffset[200:250,200:250]

from awrams.models.awral import model

m = model.AWRALModel()
imap = m.get_default_mapping()

ods = OnDemandSimulator(m,imap)

print("running...")
res = ods.run(period,extent)

from awrams.utils.io.data_mapping import SplitFileManager

print("opening comparison results")

sfm = SplitFileManager.open_existing('./test_sim_outputs','qtot*','qtot')

qtot = sfm.get_data(period,extent)

max_diff = np.max(np.abs(res['qtot'] - qtot))
print(max_diff)
assert(max_diff < 1e-5)