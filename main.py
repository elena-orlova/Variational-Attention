import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import glob
import os
import sys
import random

import tools
import tools.io
import tools.Models
import tools.ModelConstructor
import tools.modules
from tools.Utils import use_gpu
import tools.opts

from utils import train_model, check_save_model_path, tally_parameters, lazily_load_dataset, load_fields, collect_report_features, build_model, build_optim

parser = argparse.ArgumentParser(
    description='main.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# tools.opts.py
tools.opts.add_md_help_argument(parser)
tools.opts.model_opts(parser)
tools.opts.train_opts(parser)

opt = parser.parse_args()

if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if len(opt.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported.\n")
    sys.exit(1)

# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient

    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(
        opt.tensorboard_log_dir + datetime.now().strftime("/%b-%d_%H-%M-%S"),
        comment="Onmt")

progress_step = 0



def main():
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = checkpoint['epoch'] + 1
    elif opt.init_with:
        print('Loading checkpoint from %s' % opt.init_with)
        checkpoint = torch.load(opt.init_with,
                                map_location=lambda storage, loc: storage)
        model_opt = opt
    elif opt.eval_with:
        print('Loading checkpoint from %s' % opt.eval_with)
        checkpoint = torch.load(opt.eval_with,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint["opt"]
        model_opt.eval_only = 1
    else:
        checkpoint = None
        model_opt = opt

    for k, v in vars(model_opt).items():
        print("{}: {}".format(k, v))


    first_dataset = next(lazily_load_dataset("train"))
    data_type = first_dataset.data_type

    fields = load_fields(first_dataset, data_type, checkpoint)

    collect_report_features(fields)

    model = build_model(model_opt, opt, fields, checkpoint)

    tally_parameters(model)
    check_save_model_path()

    optim = build_optim(model, checkpoint)

    train_model(model, fields, optim, data_type, model_opt)

    if opt.tensorboard:
        writer.close()


if __name__ == "__main__":
    main()
