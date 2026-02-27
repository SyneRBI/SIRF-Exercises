class MergeBlockDataContainers(op.LinearOperator):

    def __init__(self, domain_geometry, weights=None, **kwargs):
        range_list = []
        for geometry in domain_geometry.geometries:
            range_list.extend(geometry.geometries)
        if weights is None:
            self.weights = [1] * len(range_list)
        else:
            self.weights = weights
        super(MergeBlockDataContainers, self).__init__(domain_geometry=domain_geometry, 
                                                       range_geometry=cil.BlockGeometry(*range_list), **kwargs)

    def direct(self, x, out=None):
        res_list = []
        for i, container in enumerate(x.containers):
            tmp_container = container.clone()
            tmp_container *= self.weights[i]
            res_list.extend(tmp_container.containers)
        if out is None:
            return cil.BlockDataContainer(*res_list)
        out.fill(cil.BlockDataContainer(*res_list))

    def adjoint(self, y, out=None):
        input_list = []
        start_idx = 0
        for i, container in enumerate(self.domain_geometry().allocate().containers):
            num_containers = len(container.containers)
            sub_containers = y.containers[start_idx:start_idx + num_containers]
            input_list.append(cil.BlockDataContainer(*sub_containers)/self.weights[i])
            start_idx += num_containers
        
        if out is None:
            return cil.BlockDataContainer(*input_list)
        else:
            out.fill(cil.BlockDataContainer(*input_list))