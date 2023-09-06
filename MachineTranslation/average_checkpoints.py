# pylint: disable=C0103
"""average the checkpoint in different steps"""

import os
from collections import OrderedDict
from mindspore import save_checkpoint, load_checkpoint

source_folder = "./"  # folder containing checkpoints that need to be averaged
starts_with = "step"  # checkpoints' names begin with this string
ends_with = ".ckpt"  # checkpoints' names end with this string

# Get list of checkpoint names
checkpoint_names = [f for f in os.listdir(source_folder) if f.startswith(starts_with) and f.endswith(ends_with)]
assert len(checkpoint_names) > 0, "Did not find any checkpoints!"

# Average parameters from checkpoints
averaged_params_dict = OrderedDict()
for c in checkpoint_names:
    params_dict = load_checkpoint(c)
    param_names = params_dict.keys()
    for param_name in param_names:
        if param_name not in averaged_params_dict:
            averaged_params_dict[param_name] = params_dict[param_name].copy() * 1 / len(checkpoint_names)
        else:
            averaged_params_dict[param_name] += params_dict[param_name] * 1 / len(checkpoint_names)

# Use one of the checkpoints as a surrogate to load the averaged parameters into
params_dict = load_checkpoint(checkpoint_names[0])
params_list = []
for param_name, param_data in averaged_params_dict.items():
    assert param_name in params_dict
    params_list.append({'name': param_name, 'data': param_data})

# Save averaged checkpoint
save_checkpoint(params_list, 'avaraged_transformer.ckpt')
