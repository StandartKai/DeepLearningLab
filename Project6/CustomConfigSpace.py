import ConfigSpace as CS

def giveConfigSpace():
    cs = CS.ConfigurationSpace()


    Adam_final_lr_fraction = CS.UniformFloatHyperparameter('Adam_final_lr_fraction',
                                           lower=1e-4,
                                           upper=1.0,
                                           default_value=1e-2,
                                           log=True)

    cs.add_hyperparameter(Adam_final_lr_fraction)


    Adam_initial_lr = CS.UniformFloatHyperparameter('Adam_initial_lr',
                                           lower=1e-4,
                                           upper=1e-2,
                                           default_value=1e-3,
                                           log=True)

    cs.add_hyperparameter(Adam_initial_lr)


    SGD_final_lr_fraction = CS.UniformFloatHyperparameter('SGD_final_lr_fraction',
                                           lower=1e-4,
                                           upper=1.0,
                                           default_value=1e-2,
                                           log=True)

    cs.add_hyperparameter(SGD_final_lr_fraction)


    SGD_initial_lr = CS.UniformFloatHyperparameter('SGD_initial_lr',
                                           lower=1e-3,
                                           upper=0.5,
                                           default_value=1e-1,
                                           log=True)

    cs.add_hyperparameter(SGD_initial_lr)


    SGD_momentum = CS.UniformFloatHyperparameter('SGD_momentum',
                                           lower=0.0,
                                           upper=0.99,
                                           default_value=0.9,
                                           log=False)

    cs.add_hyperparameter(SGD_momentum)


    StepDecay_epochs_per_step = CS.UniformIntegerHyperparameter("StepDecay_epochs_per_step",
                                                   lower=1,
                                                   default_value=16,
                                                   upper=128,
                                                   log=True)

    cs.add_hyperparameter(StepDecay_epochs_per_step)


    activation = CS.CategoricalHyperparameter("activation", ['relu', 'tanh'], default_value='relu')

    cs.add_hyperparameter(activation)


    batch_size = CS.UniformIntegerHyperparameter("batch_size",
                                                   lower=8,
                                                   default_value=16,
                                                   upper=256,
                                                   log=True)

    cs.add_hyperparameter(batch_size)


    dropout_0 = CS.UniformFloatHyperparameter('dropout_0',
                                           lower=0.0,
                                           upper=0.5,
                                           default_value=0.0,
                                           log=False)

    cs.add_hyperparameter(dropout_0)


    dropout_1 = CS.UniformFloatHyperparameter('dropout_1',
                                           lower=0.0,
                                           upper=0.5,
                                           default_value=0.0,
                                           log=False)

    cs.add_hyperparameter(dropout_1)


    dropout_2 = CS.UniformFloatHyperparameter('dropout_2',
                                           lower=0.0,
                                           upper=0.5,
                                           default_value=0.0,
                                           log=False)

    cs.add_hyperparameter(dropout_2)


    dropout_3 = CS.UniformFloatHyperparameter('dropout_3',
                                           lower=0.0,
                                           upper=0.5,
                                           default_value=0.0,
                                           log=False)

    cs.add_hyperparameter(dropout_3)


    l2_reg_0 = CS.UniformFloatHyperparameter('l2_reg_0',
                                           lower=1e-6,
                                           upper=1e-2,
                                           default_value=1e-4,
                                           log=True)

    cs.add_hyperparameter(l2_reg_0)


    l2_reg_1 = CS.UniformFloatHyperparameter('l2_reg_1',
                                           lower=1e-6,
                                           upper=1e-2,
                                           default_value=1e-4,
                                           log=True)

    cs.add_hyperparameter(l2_reg_1)


    l2_reg_2 = CS.UniformFloatHyperparameter('l2_reg_2',
                                           lower=1e-6,
                                           upper=1e-2,
                                           default_value=1e-4,
                                           log=True)

    cs.add_hyperparameter(l2_reg_2)


    l2_reg_3 = CS.UniformFloatHyperparameter('l2_reg_3',
                                           lower=1e-6,
                                           upper=1e-2,
                                           default_value=1e-4,
                                           log=True)

    cs.add_hyperparameter(l2_reg_3)


    learning_rate_schedule = CS.CategoricalHyperparameter("learning_rate_schedule", ['ExponentialDecay', 'StepDecay'], default_value='ExponentialDecay')
    cs.add_hyperparameter(learning_rate_schedule)


    loss_function = CS.CategoricalHyperparameter("loss_function", ['categorical crossentropy'], default_value='categorical crossentropy')
    cs.add_hyperparameter(loss_function)


    num_layers = CS.UniformIntegerHyperparameter('num_layers',
                                             lower=1,
                                             upper=4,
                                             default_value=2,
                                             log=False)

    cs.add_hyperparameter(num_layers)


    num_units_0 = CS.UniformIntegerHyperparameter('num_units_0',
                                             lower=16,
                                             upper=256,
                                             default_value=32,
                                             log=True)

    cs.add_hyperparameter(num_units_0)


    num_units_1 = CS.UniformIntegerHyperparameter('num_units_1',
                                             lower=16,
                                             upper=256,
                                             default_value=32,
                                             log=True)

    cs.add_hyperparameter(num_units_1)


    num_units_2 = CS.UniformIntegerHyperparameter('num_units_2',
                                             lower=16,
                                             upper=256,
                                             default_value=32,
                                             log=True)

    cs.add_hyperparameter(num_units_2)


    num_units_3 = CS.UniformIntegerHyperparameter('num_units_3',
                                             lower=16,
                                             upper=256,
                                             default_value=32,
                                             log=True)

    cs.add_hyperparameter(num_units_3)


    optimizer = CS.CategoricalHyperparameter("optimizer", ['Adam', 'SGD'],
                                            default_value='Adam')


    cs.add_hyperparameter(optimizer)


    output_activation = CS.CategoricalHyperparameter("output_activation", ['softmax'],
                                              default_value='softmax')

    cs.add_hyperparameter(output_activation)


    cond = CS.EqualsCondition(Adam_final_lr_fraction, optimizer, 'Adam')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(Adam_initial_lr, optimizer, 'Adam')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(SGD_final_lr_fraction, optimizer, 'SGD')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(SGD_initial_lr, optimizer, 'SGD')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(SGD_momentum, optimizer, 'SGD')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(StepDecay_epochs_per_step, learning_rate_schedule, 'StepDecay')
    cs.add_condition(cond)


    # Add condition that the hyperparameter float_hp is only active if discrete_hp has an higher or equal value than 2

    drop = CS.GreaterThanCondition(dropout_1, num_layers, 1)
    cs.add_condition(cond)

    drop = CS.GreaterThanCondition(dropout_2, num_layers, 2)
    cs.add_condition(cond)

    drop = CS.GreaterThanCondition(dropout_3, num_layers, 3)
    cs.add_condition(cond)


    drop = CS.GreaterThanCondition(l2_reg_1, num_layers, 1)
    cs.add_condition(cond)

    drop = CS.GreaterThanCondition(l2_reg_2, num_layers, 2)
    cs.add_condition(cond)

    drop = CS.GreaterThanCondition(l2_reg_3, num_layers, 3)
    cs.add_condition(cond)


    drop = CS.GreaterThanCondition(num_units_1, num_layers, 1)
    cs.add_condition(cond)

    drop = CS.GreaterThanCondition(num_units_2, num_layers, 2)
    cs.add_condition(cond)

    drop = CS.GreaterThanCondition(num_units_3, num_layers, 3)
    cs.add_condition(cond)
    return cs
