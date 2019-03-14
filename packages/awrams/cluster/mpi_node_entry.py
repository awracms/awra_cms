def run_from_pickle(pklfile):
    import pickle
    from mpi4py import MPI
    import time

    cspec = pickle.load(open(pklfile,'rb'))

    rank = MPI.COMM_WORLD.rank

    node_spec = cspec.node_map[rank]

    node = node_spec.instantiate(MPI)

    comm_world = MPI.COMM_WORLD
    node.comm_world = comm_world
    g_world = comm_world.Get_group()

    for cname, contract in cspec.data_contracts.items():
        g_contract = g_world.Incl(contract.src_group.members + contract.dest_group.members)
        comm_msg = comm_world.Create(g_contract)
        comm_data = comm_world.Create(g_contract)
        if node_spec.group in [contract.src_group,contract.dest_group]:
            if node_spec.group == contract.src_group:
                node.out_contracts[cname] = contract
            else:
                node.in_contracts[cname] = contract
            node.comms_msg[cname] = comm_msg # messaging communicators.  separate messages for handshaking/ready/etc?
            # readiness can be expressed as a barrier;  handshaking needs message content
            node.comms_data[cname] = comm_data # dataflow communicators
            node.src_members[cname] = list(range(contract.src_group.size))
            node.dest_members[cname] = list(range(contract.src_group.size,contract.src_group.size + contract.dest_group.size))

    comm_world.barrier()

    #node._build_communicators()
    #node._init_msg_reqs()
    node.initialise()

    #node.tasks = [0,1] #+++ hardcode to get running
    #node.tasks.reverse()

    node.run()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Launch a clustered job')
    parser.add_argument('pickle_file', type=str,
                        help='filename of pickled cal_spec')

    args = parser.parse_args()

    pklfile = args.pickle_file

    run_from_pickle(pklfile)
