import logging
import os, shutil

import matplotlib.pyplot as plt
import torch

# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(os.path.join('results',path))

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join('results',path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join('results',path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def save_or_show_plot(file_nm: str, save: bool):
    if save:
        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", file_nm))
    else:
        plt.show()


def numpy_to_tvar(x):
    return torch.from_numpy(x).type(torch.FloatTensor).cuda()
