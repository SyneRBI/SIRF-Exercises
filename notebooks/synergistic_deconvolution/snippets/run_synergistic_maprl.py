initial_estimate = cil.BlockDataContainer(image_dict['OSEM'], image_dict['OSEM_amyloid'])

zero_operator_im2im = op.ZeroOperator(image_dict['PET'])
zero_operator_im2grad = op.ZeroOperator(image_dict['PET'], grad_ref)

f1 = fn.OperatorCompositionFunction(fn.KullbackLeibler(b=image_dict['OSEM'], eta=image_dict['OSEM'].get_uniform_copy(1e-6)), convolve)
f2 = fn.OperatorCompositionFunction(fn.KullbackLeibler(b=image_dict['OSEM_amyloid'], eta=image_dict['OSEM_amyloid'].get_uniform_copy(1e-6)), convolve)

F = fn.BlockFunction(f1, f2)
joint_dgrad = op.BlockOperator(operator, zero_operator_im2grad, zero_operator_im2grad, operator, shape=(2,2))
G = fn.OperatorCompositionFunction(alpha* fn.SmoothMixedL21Norm(epsilon=1e-4), 
                                    op.CompositionOperator(MergeBlockDataContainers(joint_dgrad.range_geometry(),[1/6,5/6]), joint_dgrad))

maprl_synergistic = MAPRL(initial_estimate=initial_estimate, data_fidelity=F, prior=G,
                step_size=1, relaxation_eta=0.02, max_iteration=100, update_objective_interval=10)
maprl_synergistic.FOV_filter = bdc_FOV_filter()

show2D([maprl_synergistic.solution.containers[0], maprl_synergistic.solution.containers[1], image_dict['OSEM'], image_dict['OSEM_amyloid'], 
        maprl_synergistic.solution.containers[0]-image_dict['PET'], maprl_synergistic.solution.containers[1]-image_dict['PET_amyloid']], 
            title = ['Synergistic image', 'Synergistic amyloid image', 'OSEM', 'Amyloid OSEM', 'Synergistic error', 'Synergistic amyloid error'],
            origin = 'upper', num_cols = 2, fix_range=[(0,160), (0,80), (0,160), (0,80), (-160,160), (-80,80)])

plt.plot(maprl_synergistic.objective)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')