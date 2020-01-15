from copy import deepcopy
from functools import partial
import numpy as np
import lazyarray as la
from pyNN.standardmodels import cells, build_translations
from pynn_genn.standardmodels.cells import tau_to_decay, tau_to_init, genn_postsyn_defs
from pynn_genn.simulator import state
import logging
from pynn_genn.model import GeNNStandardCellType, GeNNDefinitions
from pygenn.genn_model import create_custom_neuron_class

def inv_val(val_name, **kwargs):
    return 1.0/kwargs[val_name]

def inv_tau_to_decay(val_name, **kwargs):
    return 1.0/la.exp(-state.dt / kwargs[val_name])

ADD_DVDT = bool(0)

_genn_neuron_defs = {}
_genn_postsyn_defs = {}

_genn_neuron_defs['IFAdapt'] = GeNNDefinitions(
    definitions = {
        "sim_code" : """
            $(I) = $(Isyn);
            if ($(RefracTime) <= 0.0) {
                scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);
                $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
                $(VThreshAdapt) = $(Vthresh) + ($(VThreshAdapt) - $(Vthresh))* $(DownThresh);
            }
            else {
                $(RefracTime) -= DT;
            }
        """,

        "threshold_condition_code" : "$(RefracTime) <= 0.0 && $(V) >= $(VThreshAdapt)",

        "reset_code" : """
            $(V) = $(Vreset);
            $(RefracTime) = $(TauRefrac);
            $(VThreshAdapt) += $(UpThresh)*($(Vthresh) - $(Vrest)); 
        """,

        "var_name_types" : [
            ("V", "scalar"),
            ("I", "scalar"),
            ("RefracTime", "scalar"),
            ("VThreshAdapt", "scalar"),
        ],

        "param_name_types": {
            "Rmembrane":  "scalar",  # Membrane resistance
            "ExpTC":      "scalar",  # Membrane time constant [ms]
            "Vrest":      "scalar",  # Resting membrane potential [mV]
            "Vreset":     "scalar",  # Reset voltage [mV]
            "Vthresh":    "scalar",  # Spiking threshold [mV]
            "Ioffset":    "scalar",  # Offset current
            "TauRefrac":  "scalar",
            "UpThresh":   "scalar",
            "DownThresh": "scalar",
        }
    },
    translations = (
        ("v_rest",      "Vrest"),
        ("v_reset",     "Vreset"),
        ("cm",          "Rmembrane",     "tau_m / cm", ""),
        ("tau_m",       "ExpTC",         partial(tau_to_decay, "tau_m"), None),
        ("tau_refrac",  "TauRefrac"),
        ("v_thresh",    "Vthresh"),
        ("i_offset",    "Ioffset"),
        ("v",           "V"),
        ("i",           "I"),
        ("mult_thresh", "UpThresh"),
        ("tau_thresh",  "DownThresh",    partial(tau_to_decay, "tau_thresh"), None),
        ("v_thresh_adapt",    "VThreshAdapt"),
    ),
    extra_param_values = {
        "RefracTime" : 0.0,
    })


class IF_curr_exp_i(cells.IF_curr_exp, GeNNStandardCellType):
    __doc__ = cells.IF_curr_exp.__doc__

    default_parameters = {
        'v_rest': -65.0,  # Resting membrane potential in mV.
        'cm': 1.0,  # Capacity of the membrane in nF
        'tau_m': 20.0,  # Membrane time constant in ms.
        'tau_refrac': 0.1,  # Duration of refractory period in ms.
        'tau_syn_E': 5.0,  # Decay time of excitatory synaptic current in ms.
        'tau_syn_I': 5.0,  # Decay time of inhibitory synaptic current in ms.
        'i_offset': 0.0,  # Offset current in nA
        'v_reset': -65.0,  # Reset potential after a spike in mV.
        'v_thresh': -50.0,  # Spike threshold in mV. STATIC, MIN
        'i': 0.0, #nA total input current

        ### https://www.frontiersin.org/articles/10.3389/fncom.2018.00074/full
        # 'tau_thresh': 80.0,
        # 'mult_thresh': 1.8,
        ### https://www.frontiersin.org/articles/10.3389/fncom.2018.00074/full
        'tau_thresh': 120.0,
        'mult_thresh': 1.8,
        'v_thresh_adapt': -50.0,  # Spike threshold in mV.

    }

    recordable = ['spikes', 'v', 'i', 'v_thresh_adapt']

    default_initial_values = {
        'v': -65.0,  # 'v_rest',
        'isyn_exc': 0.0,
        'isyn_inh': 0.0,
        'i': 0.0,
    }

    units = {
        'v': 'mV',
        'isyn_exc': 'nA',
        'isyn_inh': 'nA',
        'v_rest': 'mV',
        'cm': 'nF',
        'tau_m': 'ms',
        'tau_refrac': 'ms',
        'tau_syn_E': 'ms',
        'tau_syn_I': 'ms',
        'i_offset': 'nA',
        'v_reset': 'mV',
        'v_thresh': 'mV',
        'i': 'nA',
        'tau_thresh': 'ms',
        'mult_thresh': '',
        'v_thresh_adapt': 'mV',
    }

    receptor_types = (
        'excitatory', 'inhibitory',
    )

    genn_neuron_name = "IF_i"
    genn_postsyn_name = "ExpCurr"
    neuron_defs = _genn_neuron_defs['IFAdapt']
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]


if bool(ADD_DVDT):
    _genn_neuron_defs["IFSlow"] = GeNNDefinitions(
        definitions={
            "sim_code": """
                if ($(RefracTime) <= 0.0) {
                    scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);
                    $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
                    $(Vthresh) = max($(ThreshMin), $(Vthresh)*$(ThreshMultDown));
                }
                else {
                    $(RefracTime) -= DT;

                }

                $(VSlow) = (($(ExpSlow) * $(VSlow)) + ($(V) * (1.0 - $(ExpSlow))));
                $(dvdt) = ($(VSlow) - $(VSlowOld));
                $(VSlowOld) = ($(VSlow));
            """,

            "threshold_condition_code": "$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)",

            "reset_code": """
                $(V) = $(Vreset);
                $(RefracTime) = $(TauRefrac);
                $(Vthresh) = min($(ThreshMax), $(Vthresh)*$(ThreshMultUp));
            """,

            "var_name_types": [
                ("V", "scalar"),
                ("RefracTime", "scalar"),
                # Slow down
                ("VSlow", "scalar"),
                ("VSlowOld", "scalar"),
                ("dvdt", "scalar"),
                # Adaptive Threshold
                ("Vthresh", "scalar"),
            ],

            "param_name_types": {
                "Rmembrane": "scalar",  # Membrane resistance
                "ExpTC": "scalar",  # Membrane time constant [ms]
                "Vrest": "scalar",  # Resting membrane potential [mV]
                "Vreset": "scalar",  # Reset voltage [mV]
                "Vthresh": "scalar",  # Spiking threshold [mV]
                "Ioffset": "scalar",  # Offset current
                "TauRefrac": "scalar",
                # Slow down
                "ExpSlow": "scalar",
                # Adaptive Threshold
                "ThreshMultUp": "scalar",
                "ThreshMultDown": "scalar",
                "ThreshMax": "scalar",
                "ThreshMin": "scalar",
            }
        },
        translations=(
            ("v_rest", "Vrest"),
            ("v_reset", "Vreset"),
            ("cm", "Rmembrane", "tau_m / cm", ""),
            ("tau_m", "ExpTC", partial(tau_to_decay, "tau_m"), None),
            ("tau_refrac", "TauRefrac"),
            ("v_thresh", "Vthresh"),
            ("i_offset", "Ioffset"),
            ("v", "V"),
            # Slow down
            ("tau_slow", "ExpSlow", partial(tau_to_decay, "tau_slow"), None),
            ("v_slow", "VSlow"),
            ("v_slow_old", "VSlowOld"),
            ("dvdt", "dvdt"),
            # Adaptive Threshold
            ("thresh_mult_up", "ThreshMultUp", partial(inv_val, "thresh_mult_up"), None),
            ("thresh_mult_down", "ThreshMultDown", partial(inv_val, "thresh_mult_down"), None),
            ("v_thresh_max", "ThreshMax"),
            ("v_thresh_min", "ThreshMin"),

        ),
        extra_param_values={
            "RefracTime": 0.0,
        }
    )

    _genn_postsyn_defs["ExpCurrSlow"] = GeNNDefinitions(
        definitions = {
            "decay_code" : "$(inSyn)*=$(expDecay);",

            "apply_input_code" : "$(Isyn) += $(init) * $(inSyn);",

            "var_name_types" : [],
            "param_name_types" : {
                "expDecay": "scalar",
                "init": "scalar"}
        },
        translations = (
            ("tau_syn_E",  "exc_expDecay",  partial(tau_to_decay, "tau_syn_E"),   None),
            ("tau_syn_I",  "inh_expDecay",  partial(tau_to_decay, "tau_syn_I"),   None),
            ("e_rev_E_slow", "exc_slow_E"),
            ("e_rev_I_slow", "inh_slow_E"),
            ("tau_syn_E_slow", "exc_slow_expDecay", partial(tau_to_decay, "tau_syn_E_slow"), None),
            ("tau_syn_I_slow", "inh_slow_expDecay", partial(tau_to_decay, "tau_syn_I_slow"), None),
            ("v_activate_slow", "VActivateSlow"),
        ),
        extra_param_values = {
            "exc_init": partial(tau_to_init, "tau_syn_E"),
            "inh_init": partial(tau_to_init, "tau_syn_I"),
            "exc_slow_init": partial(tau_to_init, "tau_syn_E_slow"),
            "inh_slow_init": partial(tau_to_init, "tau_syn_I_slow"),

    })

    _genn_postsyn_defs["ExpCondSlow"] = GeNNDefinitions(
        definitions = {
            "decay_code" : "$(inSyn)*=$(expDecay);",

            "apply_input_code" : """
                $(Isyn) += $(init) * $(inSyn) * ($(E) - $(V));
            """,

            "var_name_types" : [],
            "param_name_types" : {
                "expDecay": "scalar",
                "init": "scalar",
                "E": "scalar"}
        },
        translations = (
            ("e_rev_E",    "exc_E"),
            ("e_rev_I",    "inh_E"),
            ("tau_syn_E",  "exc_expDecay",  partial(tau_to_decay, "tau_syn_E"),   None),
            ("tau_syn_I",  "inh_expDecay",  partial(tau_to_decay, "tau_syn_I"),   None),
            ("e_rev_E_slow",    "exc_slow_E"),
            ("e_rev_I_slow",    "inh_slow_E"),
            ("tau_syn_E_slow",  "exc_slow_expDecay",  partial(tau_to_decay, "tau_syn_E_slow"),   None),
            ("tau_syn_I_slow",  "inh_slow_expDecay",  partial(tau_to_decay, "tau_syn_I_slow"),   None),
            ("v_activate_slow", "VActivateSlow"),

        ),
        extra_param_values = {
            "exc_init": partial(tau_to_init, "tau_syn_E"),
            "inh_init": partial(tau_to_init, "tau_syn_I"),
            "exc_slow_init": partial(tau_to_init, "tau_syn_E_slow"),
            "inh_slow_init": partial(tau_to_init, "tau_syn_I_slow"),
    })

    class IF_curr_exp_slow(cells.IF_curr_exp, GeNNStandardCellType):
        __doc__ = cells.IF_curr_exp.__doc__

        default_parameters = {
            'v_rest':   -65.0,  # Resting membrane potential in mV.
            'cm':         1.0,  # Capacity of the membrane in nF
            'tau_m':     20.0,  # Membrane time constant in ms.
            'tau_refrac': 0.1,  # Duration of refractory period in ms.
            'tau_syn_E':  5.0,  # Decay time of excitatory synaptic current in ms.
            'tau_syn_I':  5.0,  # Decay time of inhibitory synaptic current in ms.
            'i_offset':   0.0,  # Offset current in nA
            'v_reset':  -65.0,  # Reset potential after a spike in mV.
            'v_thresh': -50.0,  # Spike threshold in mV.
            'tau_slow':   5.0,
            'thresh_mult_up': 1.0,
            'thresh_mult_down': 1.0,
            'v_thresh_max': -50.0,
            'v_thresh_min': -50.0,
            'tau_syn_E_slow': 50.0,
            'tau_syn_I_slow': 50.0,
            'v_activate_slow': -60.0,

        }
        # Slow down
        # "tau_slow": 5.0,
        # "v_slow": 0.0,
        # "v_slow_old": 0.0,
        # "dvdt": 0.0,

        recordable = ['spikes', 'v', 'v_slow', 'dvdt', 'v_thresh']

        default_initial_values = {
            'v':                -65.0,  # 'v_rest',
            'isyn_exc':           0.0,
            'isyn_inh':           0.0,
            'v_slow':           -65.0,
            'v_slow_old':       -65.0,
            'dvdt':               0.0,

        }

        units = {
            'v':          'mV',
            'isyn_exc':   'nA',
            'isyn_inh':   'nA',
            'v_rest':     'mV',
            'cm':         'nF',
            'tau_m':      'ms',
            'tau_refrac': 'ms',
            'tau_syn_E':  'ms',
            'tau_syn_I':  'ms',
            'i_offset':   'nA',
            'v_reset':    'mV',
            'v_thresh':   'mV',
            'tau_slow':   'ms',
            'v_slow':     'mV',
            'v_slow_old': 'mV',
            'dvdt':       'mV/ms',
            # "thresh_mult_up": '',
            # "thresh_mult_down": '',
            'v_thresh_max': 'mV',
            'v_thresh_min': 'mV',
            'tau_syn_E_slow': 'ms',
            'tau_syn_I_slow': 'ms',
            'v_activate_slow': 'mV',

        }
        receptor_types = ('excitatory', 'inhibitory',
                          'excitatory_slow', 'inhibitory_slow',
                         )

        genn_neuron_name = "IFSlow"
        genn_postsyn_name = "ExpCurrSlow"
        neuron_defs = _genn_neuron_defs[genn_neuron_name]
        postsyn_defs = _genn_postsyn_defs[genn_postsyn_name]


    class IF_cond_exp_slow(cells.IF_cond_exp, GeNNStandardCellType):
        default_parameters = {
            'v_rest':          -65.0,  # Resting membrane potential in mV.
            'cm':                1.0,  # Capacity of the membrane in nF
            'tau_m':            20.0,  # Membrane time constant in ms.
            'tau_refrac':        0.1,  # Duration of refractory period in ms.
            'tau_syn_E':         5.0,  # Decay time of the excitatory synaptic conductance in ms.
            'tau_syn_I':         5.0,  # Decay time of the inhibitory synaptic conductance in ms.
            'e_rev_E':           0.0,  # Reversal potential for excitatory input in mV
            'e_rev_I':         -70.0,  # Reversal potential for inhibitory input in mV
            'v_thresh':        -50.0,  # Spike threshold in mV.
            'v_reset':         -65.0,  # Reset potential after a spike in mV.
            'i_offset':          0.0,  # Offset current in nA
            #slow
            'tau_slow':          5.0,
            'e_rev_E_slow':      0.0,
            'e_rev_I_slow':    -70.0,
            'tau_syn_E_slow':   50.0,
            'tau_syn_I_slow':   50.0,
            'v_activate_slow': -60.0,
            'thresh_mult_up': 1.0,
            'thresh_mult_down': 1.0,
            'v_thresh_max': -50.0,
            'v_thresh_min': -50.0,

        }
        recordable = ['spikes', 'v', 'gsyn_exc', 'gsyn_inh', 'v_slow', 'dvdt', 'v_thresh']
        default_initial_values = {
            'v':          -65.0,  # 'v_rest',
            'gsyn_exc':     0.0,
            'gsyn_inh':     0.0,
            'v_slow':     -65.0,
            'v_slow_old': -65.0,
            'dvdt':         0.0,
        }
        units = {
            'v':               'mV',
            'gsyn_exc':        'uS',
            'gsyn_inh':        'uS',
            'v_rest':          'mV',
            'cm':              'nF',
            'tau_m':           'ms',
            'tau_refrac':      'ms',
            'tau_syn_E':       'ms',
            'tau_syn_I':       'ms',
            'e_rev_E':         'mV',
            'e_rev_I':         'mV',
            'v_thresh':        'mV',
            'v_reset':         'mV',
            'i_offset':        'nA',
            'tau_slow':        'ms',
            'v_slow':          'mV',
            'v_slow_old':      'mV',
            'dvdt':            'mV/ms',
            'e_rev_E_slow':    'mV',
            'e_rev_I_slow':    'mV',
            'tau_syn_E_slow':  'ms',
            'tau_syn_I_slow':  'ms',
            'v_activate_slow': 'mV',
            # "thresh_mult_up": '',
            # "thresh_mult_down": '',
            'v_thresh_max': 'mV',
            'v_thresh_min': 'mV',

        }
        receptor_types = ('excitatory', 'inhibitory',
                          'excitatory_slow', 'inhibitory_slow',
                         )
        genn_neuron_name = "IFSlow"
        genn_postsyn_name = "ExpCondSlow"
        neuron_defs = _genn_neuron_defs[genn_neuron_name]
        postsyn_defs = _genn_postsyn_defs[genn_postsyn_name]
