from nose.tools import nottest


def test_imports():
    import awrams.visualisation
    # assert False


@nottest
def test_drive():
    import awrams.visualisation.vis as vis
    import awrams.visualisation.results as res
    # from awrams.utils.catchments import CatchmentDB,CATCHMENT_SHAPEFILE
    # catchments = CatchmentDB()

    import os.path as o #import dirname,join
    import awrams.simulation as sim
    # res_dir = o.abspath(o.join('..','simulation','notebooks','_results'))
    res_dir = o.abspath(o.join(o.dirname(__file__),'..','..','simulation','notebooks','_results'))

    results = res.load_results(res_dir)
    results.path
    results.variables

    results[:,'1 dec 2010',:].spatial()

    v = results.variables.s0,results.variables.ss
    results[v,'dec 2010',vis.extents.from_boundary_coords(-39.5,143.5,-44,149)].spatial()
    vis.plt.savefig('map_of_tasmania.png', format='png', dpi=120)

    v = results.variables.qtot
    v.agg_method = 'sum'
    results[v,'dec 2010',vis.extents.from_boundary_coords(-39.5,143.5,-44,149)].spatial()

    results.variables.s0.data.shape
    results.variables.s0.agg_data.shape

    # v = results.variables.s0,results.variables.ss
    # results[v,'dec 2010',catchments.by_name.Lachlan_Gunning()].spatial(interpolation=None) #interpolation="bilinear")
    #
    # vis.show_extent(catchments.by_name.Lachlan_Gunning())
    # vis.show_extent(catchments.by_name.Lachlan_Gunning(),vis.extents.from_boundary_coords(-40,142,-30,154))
    #
    # catchments.list()

    v = results.variables.qtot,results.variables.ss
    results[v,'dec 2010',:].spatial(clim=(0,100),xlabel="longitude")

    q = results[v,'dec 2010',vis.extents.from_boundary_coords(-39.5,143.5,-44,149)]
    q.get_data_limits()

    q.spatial(clim=(0,200),xlabel="longitude")

    gridview = q.mpl
    view = gridview.children[0,1]

    view.ax.set_xlabel("ALSO LONGITUDE!")
    vis.plt.show()

    p = 'dec 2010 - jan 2011'
    e = vis.extents.from_cell_coords(-34,117)
    results[:,p,e].timeseries()

    # v = results.variables.qtot,results.variables.ss
    # p = 'dec 2010 - jan 2011'
    # e = catchments.by_name.Murrumbidgee_MittagangCrossing()
    # results[v,p,e].timeseries()
    #
    # results.variables.qtot.data.shape,results.variables.qtot.agg_data.shape

