import os
import logging
from collections import OrderedDict
import json
from datetime import datetime


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def parse(args):
    phase = args.phase
    opt_path = args.config
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['phase'] = phase


    # debug
    if 'debug' in opt['name']:
        opt['train']['val_freq'] = 2
        opt['train']['print_freq'] = 2
        opt['train']['save_checkpoint_freq'] = 3
        opt['datasets']['train']['batch_size'] = 2
        opt['model']['beta_schedule']['train']['n_timestep'] = 10
        opt['model']['beta_schedule']['val']['n_timestep'] = 10
        opt['datasets']['train']['data_len'] = 6
        opt['datasets']['val']['data_len'] = 3
    try:
        log_wandb_ckpt = args.log_wandb_ckpt
        opt['log_wandb_ckpt'] = log_wandb_ckpt
    except:
        pass
    try:
        log_eval = args.log_eval
        opt['log_eval'] = log_eval
    except:
        pass
    try:
        log_infer = args.log_infer
        opt['log_infer'] = log_infer
    except:
        pass
    
    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, '{}.log'.format(phase))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)
