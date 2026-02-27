def create_acq_model(attn_image, background_term, AC = False):
    '''create a PET acquisition model.
    
    Arguments:
    attn_image: the mu-map
    background_term: background-term as a sirf.STIR.AcquisitionData
    '''
    # create attenuation
    if AC:
        acq_model_for_attn = pet.AcquisitionModelUsingRayTracingMatrix()
        asm_attn = pet.AcquisitionSensitivityModel(attn_image, acq_model_for_attn)
        asm_attn.set_up(background_term)
        attn_factors = asm_attn.forward(background_term.get_uniform_copy(1))
        asm_attn = pet.AcquisitionSensitivityModel(attn_factors)
    
    # create acquisition model
    acq_model_w_attn = pet.AcquisitionModelUsingRayTracingMatrix()
    # we will increase the number of rays used for every Line-of-Response (LOR) as an example
    # (it is not required for the exercise of course)
    acq_model_w_attn.set_num_tangential_LORs(5)
    if AC: acq_model_w_attn.set_acquisition_sensitivity(asm_attn)
    # set-up
    acq_model_w_attn.set_up(background_term,attn_image)
    
    return acq_model_w_attn