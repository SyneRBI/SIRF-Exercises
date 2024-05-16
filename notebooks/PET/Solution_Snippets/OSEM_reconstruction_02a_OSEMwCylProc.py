def OSEM(acq_data, acq_model, initial_image, num_subiterations, num_subsets=1):
    '''run OSEM
    
    Arguments:
    acq_data: the (measured) data
    acq_model: the acquisition model
    initial_image: used for initialisation (and sets voxel-sizes etc)
    num_subiterations: number of sub-iterations (or image updates)
    num_subsets: number of subsets (defaults to 1, i.e. MLEM)
    '''
    proc = pet.TruncateToCylinderProcessor()
    proc.set_strictly_less_than_radius(True)
    proc.apply(initial_image) #, will change image
    
    obj_fun = pet.make_Poisson_loglikelihood(acq_data)
    obj_fun.set_acquisition_model(acq_model)

    recon = pet.OSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(num_subsets)
    recon.set_num_subiterations(num_subiterations)
    recon.set_current_estimate(initial_image)
    recon.set_up(initial_image)
    recon.process()
    return recon.get_output()