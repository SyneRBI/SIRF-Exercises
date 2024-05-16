data_fidelity = fn.OperatorCompositionFunction(f1, convolve)
prior = alpha * fn.OperatorCompositionFunction(fn.SmoothMixedL21Norm(epsilon=1e-4), grad)
prior_d = alpha * fn.OperatorCompositionFunction(fn.SmoothMixedL21Norm(epsilon=1e-4), operator)

maprl = MAPRL(initial_estimate=image_dict['OSEM'], data_fidelity=data_fidelity, prior=prior, 
              step_size=0.1, relaxation_eta=0.01, update_objective_interval=10)

maprl.run(verbose=1, iterations=100)

show2D([maprl.solution, image_dict['OSEM'], maprl.solution-image_dict['PET']],
         title = ['MAPRL image', 'OSEM image', 'difference to GT'], 
         origin = 'upper', num_cols = 3, fix_range=[(0,160), (0,160), (-40,40)],)

plt.plot(maprl.objective)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')