def OSEM(acquired_data, acq_model, initial_image, num_iterations):
    estimated_image=initial_image.clone()
    # some stuff here - hint, this will be similar to your solution for MLEM
    # but you will have to additionally iterate over your subsets 
    for i in range(num_iterations):
        for s in range(acq_model.num_subsets):

            subs_sensitivity = acq_model.backward(acquired_data.get_uniform_copy(1), subset_num=s)

            quotient = acquired_data/acq_model.forward(estimated_image.clone(), subset_num=s)     # y / (Ax + b)
            quotient.fill(numpy.nan_to_num(quotient.as_array()))
                
            mult_update = acq_model.backward(quotient.clone(), subset_num=s)/subs_sensitivity     # A^t * quotient / A^t1
            mult_update.fill(numpy.nan_to_num(mult_update.as_array()))
            
            estimated_image *= mult_update                                                        # update (in place)

            est_img_arr = estimated_image.as_array()
            est_img_arr[est_img_arr<0] = 0
            estimated_image.fill(numpy.nan_to_num(est_img_arr))
    
    #  some stuff here
    return estimated_image