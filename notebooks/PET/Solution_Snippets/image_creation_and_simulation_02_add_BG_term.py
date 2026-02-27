background_term = acquired_data_with_attn.get_uniform_copy(acquired_data_with_attn.max()/10)
acq_model_with_attn.set_background_term(background_term)
acquired_data_with_attn_with_bg = acq_model_with_attn.forward(image.clone())