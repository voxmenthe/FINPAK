decent_checkpoints = {

    "vMP009h_id_0_arc_v4_tc15_vc14_e2997_valloss_0.0000232.pt": {
        "fname": "saved_checkpoints/vMP009h_id_0_arc_v4_tc15_vc14_e2997_valloss_0.0000232.pt",
        "inference_parameters": {
            "rating": "very good - stays more in bounds - less directional",
            "stability_threshold": 0.02,
            "dampening_factor": 0.95,
            "ewma_alpha": 0.85,
            "temperature": 0.05,
            "return_scaling": 0.001,
            "beginning_uncertainty": 1e-05,
            "uncertainty_growth": 1e-06,
            "max_uncertainty_single": 0.03,
            "max_uncertainty_multi": 0.08,
            "uncertainty_damping": 0.99
        }
    },

    "vMP009h_id_0_arc_v4_tc15_vc14_e2689_valloss_0.0000180.pt": {
        "fname": "saved_checkpoints/vMP009h_id_0_arc_v4_tc15_vc14_e2689_valloss_0.0000180.pt",
        "inference_parameters": {
            "rating": "very good - stays more in bounds - less directional",
            "stability_threshold": 0.02,
            "dampening_factor": 0.95,
            "ewma_alpha": 0.85,
            "temperature": 0.05,
            "return_scaling": 0.05,
            "beginning_uncertainty": 1e-05,
            "uncertainty_growth": 1e-06,
            "max_uncertainty_single": 0.07,
            "max_uncertainty_multi": 0.15,
            "uncertainty_damping": 0.99
        }
    },

    "vMP009h_id_0_arc_v4_tc6_vc5_e2138_valloss_0.0004537.pt": {
        "fname": "checkpoints/vMP009h_id_0_arc_v4_tc6_vc5_e2138_valloss_0.0004537.pt",
        "inference_parameters": {
            "rating": "pretty good",
            "stability_threshold": 0.02,
            "dampening_factor": 0.95,
            "ewma_alpha": 0.85,
            "temperature": 0.35,
            "return_scaling": 0.55,
            "beginning_uncertainty": 1e-05,
            "uncertainty_growth": 1e-06,
            "max_uncertainty_single": 0.07,
            "max_uncertainty_multi": 0.15,
            "uncertainty_damping": 0.99
        }
    },

    "vMP009h_id_0_arc_v4_tc11_vc6_e522":  {
        'fname': 'saved_checkpoints/vMP009h_id_0_arc_v4_tc11_vc6_e522_valloss_0.0000474.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
            'uncertainty_damping': 0.99
        },
    }, # very good - with stab_thresh 0.02, damp_factor 0.95, ewma 0.85, temp 0.65, returnscale 0.55, bu 0.00001, ug 0.000001, mus 0.07, mum 0.15, ud 0.95
    
    "vMP009h_id_0_arc_v4_tc1_vc13_e318":  {
        'fname': 'saved_checkpoints/vMP009h_id_0_arc_v4_tc1_vc13_e318_valloss_0.0001148.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
    }, # excellent with stability_threshold 0.02, dampening_factor 0.95, ewma_alpha 0.85, temp 0.45, return scaling 0.75, bu 0.0001, ug 0.00001, mus 0.07, mum 0.14, ud 0.95
    
    "vMP009h_id_0_arc_v4_tc12_vc12_e305":  {
        'fname': 'saved_checkpoints/vMP009h_id_0_arc_v4_tc12_vc12_e305_valloss_0.0027858.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
    }, # excellent with stability_threshold 0.02, dampening_factor 0.95, ewma_alpha 0.85, temp 0.45, return scaling 0.75, bu 0.0001, ug 0.00001, mus 0.07, mum 0.14, ud 0.95
    
    "vMP009h_id_0_arc_v4_tc10_vc10_e259":  {
        'fname': 'checkpoints/vMP009h_id_0_arc_v4_tc10_vc10_e259_valloss_0.0019606.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
    }, # v4 - ok fairly different between runs -- stability 0.02, dampening 0.95, ewma 0.85, temp 0.7 - bu 0.0001, ug 0.00001, mus 0.07, mum 0.14, udm 0.95 + 20 different starting points per averaged index
    
    "vMP009h_id_0_arc_v4_tc11_vc11_e279":  {
        'fname': 'checkpoints/vMP009h_id_0_arc_v4_tc11_vc11_e279_valloss_0.0000458.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
    }, # v4 - ok does 4/10-ish -- stability 0.02, dampening 0.95, ewma 0.85, temp 0.7 - bu 0.0001, ug 0.00001, mus 0.07, mum 0.14, udm 0.95 + 20 different starting points per averaged index
    
    "vMP009h_id_0_arc_v4_tc8_vc8_e200":  {
        'fname': 'saved_checkpoints/vMP009h_id_0_arc_v4_tc8_vc8_e200_valloss_0.0147256.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
    }, # v4 - pretty good -- stability 0.02, dampening 0.9, ewma 0.65, temp 0.01 - bu 0.0001, ug 0.00001, mus 0.09, mum 0.16, udm 0.9
    
    "vMP009a_id_0_arc_v3_tc11_vc5_e1710":  {
        'fname': 'saved_checkpoints/vMP009a_id_0_arc_v3_tc11_vc5_e1710_valloss_0.0013949.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
    }, # v3 - actually pretty decent - stability 0.02, dampening 0.9, ewma 0.65, temp 0.01 - bu 0.0001, ug 0.00001, mus 0.09, mum 0.16, udm 0.9
    
    "vMP009a_id_0_arc_v3_tc11_vc5_e1710":  {
        'fname': 'saved_checkpoints/vMP009a_id_0_arc_v3_tc11_vc5_e1710_valloss_0.0013949.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
    }, # v3 - the predictions are excellent - work on the bounds tho
    
    "vMP003h_id_0_arc_v3_tc11_vc5_e668":  {
        'fname': 'saved_checkpoints/vMP003h_id_0_arc_v3_tc11_vc5_e668_valloss_0.0099780.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
    }, # v4 - predictions seem excellent - need to work on the bounds tho
    
    "vMP005a_id_1_arc_v3_e778":  {
        'fname': 'saved_checkpoints/vMP005a_final_id_1_arc_v3_e778_valloss_0.0002237.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
    }, # # good. stability 0.02, dampening 0.9, ewma 0.65, temp 0.01, return scaling 0.5
    
    "vMP003d_id_0_arc_v3_tc6_vc2_e338":  {
        'fname': 'saved_checkpoints/vMP003d_id_0_arc_v3_tc6_vc2_e338_valloss_0.0020710.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
    }, # excellent with stability_threshold 0.02, dampening_factor 0.9, ewma_alpha 0.65, temp 0.01# 
    
    "vMP003d_id_0_arc_v3_tc7_vc3_e353":  {
        'fname': 'saved_checkpoints/vMP003d_id_0_arc_v3_tc7_vc3_e353_valloss_0.0021448.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
    }, # excellent with stability_threshold 0.02, dampening_factor 0.9, ewma_alpha 0.65, temp 0.01
    
    "vMP003c_id_0_arc_v3_tc19_vc8_e411":  {
        'fname': 'saved_checkpoints/vMP003c_id_0_arc_v3_tc19_vc8_e411_valloss_0.0024866.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
    }, # good with stability_threshold 0.02, dampening_factor 0.9, ewma_alpha 0.65, temp 0.01
    
    "vMP003c_id_0_arc_v3_tc20_vc9_e425":  {
        'fname': 'saved_checkpoints/vMP003c_id_0_arc_v3_tc20_vc9_e425_valloss_0.0053009.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
    }, # good with stability_threshold 0.02, dampening_factor 0.9, ewma_alpha 0.65, temp 0.01
    
    "vMP003b_id_1_arc_v3_tc9_vc7_e441":  {
        'fname': 'saved_checkpoints/vMP003b_id_1_arc_v3_tc9_vc7_e441_valloss_0.0005869.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
        }, # seems good with stability_threshold 0.02, dampening_factor 0.9, ewma_alpha 0.65, temp 0.01
    "vMP003b_id_0_arc_v3_tc7_vc6_e326":  {
        'fname': 'saved_checkpoints/vMP003b_id_0_arc_v3_tc7_vc6_e326_valloss_0.0003369.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
        }, # excellent with stability_threshold 0.02, dampening_factor 0.9, ewma_alpha 0.65, temp 0.01
    "vMP003a_id_0_arc_v3_tc8_vc4_e413":  {
        'fname': 'saved_checkpoints/vMP003a_id_0_arc_v3_tc8_vc4_e413_valloss_0.0009714.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
        }, # input features only 10?
    "vMP003b_id_0_arc_v3_tc4_vc3_e252":  {
        'fname': 'saved_checkpoints/vMP003b_id_0_arc_v3_tc4_vc3_e252_valloss_0.0021478.pt',
        'inference_parameters': {
            'stability_threshold': 0.02,,
            'dampening_factor': 0.95,
            'ewma_alpha': 0.85,
            'temperature': 0.65,
            'return_scaling': 0.55,
            'beginning_uncertainty': 0.0001,
            'uncertainty_growth': 0.00001,
            'max_uncertainty_single': 0.07,
            'max_uncertainty_multi': 0.14,
        },
        }, # ok with multihorizon
}