# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from widgbed.lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.
    if algorithm == "ADRMX":
        _hparam('lr', 3e-5, lambda r: r.choice([2e-5, 3e-5, 4e-5, 5e-5]))
    else:
        _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))
    _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))
    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet50_augmix', True, lambda r: True)
    _hparam('dinov2', False, lambda r: False)
    _hparam('vit', False, lambda r: False)
    _hparam('vit_attn_tune', False, lambda r: False)
    _hparam('freeze_bn', False, lambda r: False)
    _hparam('lars', False, lambda r: False)
    _hparam('linear_steps', 500, lambda r: 500)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('vit_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if 'DANN' in algorithm:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5))
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5))
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2))

    elif algorithm == 'Fish':
        _hparam('meta_lr', 0.5, lambda r:r.choice([0.05, 0.1, 0.5]))

    elif algorithm == "RDM": 
        _hparam('rdm_lambda', 5.0, lambda r: r.uniform(0.1, 10.0))
        _hparam('rdm_penalty_anneal_iters', 1500, lambda r: int(r.uniform(800, 2700)))    
        _hparam('variance_weight', 0.004, lambda r: r.uniform(0.001, 0.007))  
        _hparam('rdm_lr', 1.5e-5, lambda r: r.uniform(8e-6, 2e-5))

    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif "Mixup" in algorithm:
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, 1))
        
    elif algorithm == 'RandomErasing':
        _hparam("probability", 0.5, lambda r: r.choice([0., 0.5])) #将执行操作的概率。The probability that the operation will be performed.
        _hparam("sl",0.02, lambda r: 10 ** r.uniform(-1, 1)) #min erasing area
        _hparam("sh",0.4 , lambda r: 10 ** r.uniform(-1, 1))#max erasing area
        _hparam("r1",0.3, lambda r: 10**r.uniform(-1, 1)) #最小纵横比
        
    elif algorithm == "CutMix":
        hparams["beta"] = (1.0, 1.0)
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        hparams["cutmix_prob"] = (1.0, 1.0)

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm in [ "MMD","CORAL","CausIRLCORAL","CausIRLMMD"]:
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))

    elif "MLDG" in algorithm:
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))
        _hparam('n_meta_test', 2, lambda r:  r.choice([1, 2]))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -1))

    elif algorithm == "ANDMask":
        _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

    elif algorithm == "IGA":
        _hparam('penalty', 1000, lambda r: 10**r.uniform(1, 5))

    elif algorithm == "SANDMask":
        _hparam('tau', 1.0, lambda r: r.uniform(0.0, 1.))
        _hparam('k', 1e+1, lambda r: 10**r.uniform(-3, 5))

    elif algorithm == "Fishr":
        _hparam('lambda', 1000., lambda r: 10**r.uniform(1., 4.))
        _hparam('penalty_anneal_iters', 1500, lambda r: int(r.uniform(0., 5000.)))
        _hparam('ema', 0.95, lambda r: r.uniform(0.90, 0.99))

    elif algorithm == "TRM":
        _hparam('cos_lambda', 1e-4, lambda r: 10 ** r.uniform(-5, 0))
        _hparam('iters', 200, lambda r: int(10 ** r.uniform(0, 4)))
        _hparam('groupdro_eta', 1e-2, lambda r: 10 ** r.uniform(-3, -1))

    elif algorithm == "IBERM":
        _hparam('ib_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "IBIRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))
        _hparam('ib_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "CAD" or algorithm == "CondCAD":
        _hparam('lmbda', 1e-1, lambda r: r.choice([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
        _hparam('temperature', 0.1, lambda r: r.choice([0.05, 0.1]))
        _hparam('is_normalized', False, lambda r: False)
        _hparam('is_project', False, lambda r: False)
        _hparam('is_flipped', True, lambda r: True)

    elif algorithm == "Transfer":
        _hparam('t_lambda', 1.0, lambda r: 10**r.uniform(-2, 1))
        _hparam('delta', 2.0, lambda r: r.uniform(0.1, 3.0))
        _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('gda', False, lambda r: True)
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))

    elif algorithm == 'EQRM':
        _hparam('eqrm_quantile', 0.75, lambda r: r.uniform(0.5, 0.99))
        _hparam('eqrm_burnin_iters', 2500, lambda r: 10 ** r.uniform(2.5, 3.5))
        _hparam('eqrm_lr', 1e-6, lambda r: 10 ** r.uniform(-7, -5))

    elif algorithm == 'ERMPlusPlus':
        _hparam('linear_lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    elif algorithm == 'URM':
        _hparam('urm', 'adversarial', lambda r: str(r.choice(['adversarial']))) # 'adversarial'
        
        _hparam('urm_adv_lambda', 0.1, lambda r: float(r.uniform(0,0.2)))
        _hparam('urm_discriminator_label_smoothing', 0, lambda r: float(r.uniform(0, 0)))
        _hparam('urm_discriminator_optimizer', 'adam', lambda r: str(r.choice(['adam'])))
        _hparam('urm_discriminator_hidden_layers', 1, lambda r: int(r.choice([1,2,3])))
        _hparam('urm_generator_output', 'tanh', lambda r: str(r.choice(['tanh', 'relu'])))
        _hparam('urm_discriminator_lr', 5e-5, lambda r: 10**r.uniform(-6, -4.5))
        
    elif algorithm == 'SAM':
        _hparam("rho", 0.05, lambda r: r.choice([0.01, 0.02, 0.05, 0.1]))
        
    elif algorithm == "ADRMX":
        _hparam('cnt_lambda', 1.0, lambda r: r.choice([1.0]))
        _hparam('dclf_lambda', 1.0, lambda r: r.choice([1.0]))
        _hparam('disc_lambda', 0.75, lambda r: r.choice([0.75]))
        _hparam('rmxd_lambda', 1.0, lambda r: r.choice([1.0]))
        _hparam('d_steps_per_g_step', 2, lambda r: r.choice([2]))
        _hparam('beta1', 0.5, lambda r: r.choice([0.5]))
        _hparam('mlp_width', 256, lambda r: r.choice([256]))
        _hparam('mlp_depth', 9, lambda r: int(r.choice([8, 9, 10])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0]))
    
    elif algorithm == 'LIRR':
        _hparam("lambda",1.0,   lambda r: 10 ** r.uniform(-2, 2))
        _hparam("lambda_inv_risks",100,   lambda r: int(10**r.uniform(0, 2)))
        _hparam("d_steps_per_g_step", 1,   lambda r: int(2 ** r.uniform(0, 3)))
        _hparam("grad_penalty",0.,   lambda r: 10 ** r.uniform(-2, 1))
        _hparam("mlp_width",256,  lambda r: int(2 ** r.uniform(6, 10)))
        _hparam("mlp_depth",3,   lambda r: int(r.choice([3, 4, 5])))
        _hparam("mlp_dropout",0.,   lambda r: r.choice([0., 0.1, 0.5]))

    elif algorithm == 'IIB':
        _hparam("lambda_beta",1e-4,   lambda r: 1e-3 ** r.uniform(0, 2))
        _hparam("lambda_inv_risks", 10,   lambda r: int(10 ** r.uniform(0, 2)))
        _hparam("enable_bn", True,   lambda r: bool(r.choice([True, True])))
        
    elif 'CrossGrad' in algorithm:
        _hparam("eps_f", 1.0,   lambda r: 10 ** r.uniform(-2, 2))
        _hparam("eps_d",1.0,   lambda r: 10 ** r.uniform(-2, 2))
        _hparam("alpha_f", 0.5, lambda r: r.choice([0., 0.5]))
        _hparam("alpha_d", 0.5, lambda r: r.choice([0., 0.5]))

    elif 'MMDAAE' in algorithm:
        _hparam("w_mmd", 2.0,   lambda r: 10 ** r.uniform(-2, 2))
        _hparam("w_adv", 0.1, lambda r: r.choice([0., 0.5]))
        _hparam("w_ae", 0.1, lambda r: r.choice([0., 0.5]))
        _hparam("w_cls", 1.0,   lambda r: 10 ** r.uniform(-2, 2))

    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.
    return hparams


def default_hparams(algorithm, NIC):
    return {a: b for a, (b, c) in _hparams(algorithm, NIC, 0).items()}


def random_hparams(algorithm, NIC, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, NIC, seed).items()}
