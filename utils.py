import argparse
import glob
import os
import sys
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import tools
import tools.io
import tools.Models
import tools.ModelConstructor
import tools.modules
from tools.Utils import use_gpu
import tools.opts

parser = argparse.ArgumentParser(
    description='main.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# tools.opts.py
tools.opts.add_md_help_argument(parser)
tools.opts.model_opts(parser)
tools.opts.train_opts(parser)

opt = parser.parse_args()


def report_func(epoch, batch, num_batches,
                progress_step,
                start_time, lr, report_stats):
    """
    This is batch-level traing progress report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        progress_step(int): the progress step.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        if opt.tensorboard:
            # Log the progress using the number of batches on the x-axis.
            report_stats.log_tensorboard(
                "progress", writer, lr, progress_step)
        report_stats = tools.Statistics()

    return report_stats


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train, random_state=None):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train
        self.random_state = random_state

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            if self.random_state is not None:
                self.cur_iter.random_shuffler._random_state = self.random_state
            for batch in self.cur_iter:
                yield batch
            self.random_state = self.cur_iter.random_shuffler._random_state
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return tools.io.OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort=False, sort_within_batch=True,
            repeat=False)


def make_dataset_iter(datasets, fields, opt, is_train=True, random_state=None):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        # In token batching scheme, the number of sequences is limited
        # such that the total number of src/tgt tokens (including padding)
        # in a batch <= batch_size
        def batch_size_fn(new, count, sofar):
            # Maintains the longest src and tgt length in the current batch
            global max_src_in_batch, max_tgt_in_batch
            # Reset current longest length at a new batch (count=1)
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            # Src: <bos> w1 ... wN <eos>
            max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
            # Tgt: w1 ... wN <eos>
            max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)

    device = 'cuda' if opt.gpuid else 'cpu'

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train, random_state=random_state)


def make_loss_compute(model, tgt_vocab, opt, train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute = tools.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength)
    else:
        compute = tools.Loss.NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing if train else 0.0,
            train_baseline=opt.train_baseline > 0,
        )

    if use_gpu(opt):
        compute.cuda()

    return compute


def train_model(model, fields, optim, data_type, model_opt):
    train_loss = make_loss_compute(model, fields["tgt"].vocab, opt)
    valid_loss = make_loss_compute(model, fields["tgt"].vocab, opt,
                                   train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count

    trainer = tools.Trainer(model, train_loss, valid_loss, optim,
                            trunc_size, shard_size, data_type,
                            norm_method, grad_accum_count)

    if model_opt.eval_only > 0:
        print("|Param|: {}".format(sum([p.norm()**2 for p in model.parameters()]).data[0]**0.5))
        print("ELBO_q")
        model.use_prior = False
        valid_iter = make_dataset_iter(lazily_load_dataset("valid"),
                                       fields, opt,
                                       is_train=False)
        valid_stats = trainer.validate(valid_iter, "enum")
        print('Validation exp(elbo): %g' % valid_stats.expelbo())
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation xent: %g' % valid_stats.xent())
        print('Validation kl: %g' % valid_stats.kl())
        print('Validation accuracy: %g' % valid_stats.accuracy())
        print("N validation words: {}".format(valid_stats._n_words))
        print("p(x)")
        model.use_prior = True
        for k in range(6):
            print("k-max: {}".format(k))
            model.k = k
            valid_iter = make_dataset_iter(lazily_load_dataset("valid"),
                                           fields, opt,
                                           is_train=False)
            valid_stats = trainer.validate(valid_iter, "exact")
            print('Validation exp(elbo): %g' % valid_stats.expelbo())
            print('Validation perplexity: %g' % valid_stats.ppl())
            print('Validation xent: %g' % valid_stats.xent())
            print('Validation kl: %g' % valid_stats.kl())
            print('Validation accuracy: %g' % valid_stats.accuracy())
            print("N validation words: {}".format(valid_stats._n_words))
        return 0


    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    print(' * batch size: %d' % opt.batch_size)
    random_state = None
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        train_iter = make_dataset_iter(lazily_load_dataset("train"),
                                       fields, opt, random_state=random_state)
        train_stats = trainer.train(train_iter, epoch, report_func)
        random_state = train_iter.random_state
        print('Train exp(elbo): %g' % train_stats.expelbo())
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train xent: %g' % train_stats.xent())
        print('Train kl: %g' % train_stats.kl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_iter = make_dataset_iter(lazily_load_dataset("valid"),
                                       fields, opt,
                                       is_train=False)
        if model.mode == 'sample' or model.mode == 'wsram' or model.mode == 'gumbel':
            val_mode = 'enum'
        else:
            val_mode = model.mode
        valid_stats = trainer.validate(valid_iter, val_mode)
        print('Validation exp(elbo): %g' % valid_stats.expelbo())
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation xent: %g' % valid_stats.xent())
        print('Validation kl: %g' % valid_stats.kl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)
        if opt.tensorboard:
            train_stats.log_tensorboard("train", writer, optim.lr, epoch)
            train_stats.log_tensorboard("valid", writer, optim.lr, epoch)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.expelbo(), epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats)


def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def lazily_load_dataset(corpus_type):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        print('!!!!', pt_file)
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one tools.io.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield lazy_dataset_loader(pt, corpus_type)


def load_fields(dataset, data_type, checkpoint):
    if checkpoint is not None:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = tools.io.load_fields_from_vocab(
            checkpoint['vocab'], data_type)
    else:
        fields = tools.io.load_fields_from_vocab(
            torch.load(opt.data + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    if data_type == 'text':
        print(' * vocabulary size. source = %d; target = %d' %
              (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        print(' * vocabulary size. target = %d' %
              (len(fields['tgt'].vocab)))

    return fields


def collect_report_features(fields):
    src_features = tools.io.collect_features(fields, side='src')
    tgt_features = tools.io.collect_features(fields, side='tgt')

    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        print(' * tgt feature %d size = %d' % (j, len(fields[feat].vocab)))


def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = tools.ModelConstructor.make_base_model(model_opt, fields,
                                                   use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model


def build_optim(model, checkpoint):
    saved_optimizer_state_dict = None

    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        # We need to save a copy of optim.optimizer.state_dict() for setting
        # the, optimizer state later on in Stage 2 in this method, since
        # the method optim.set_parameters(model.parameters()) will overwrite
        # optim.optimizer, and with ith the values stored in
        # optim.optimizer.state_dict()
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        print('Making optimizer for training.')
        optim = tools.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            eps=opt.adam_eps,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=None)

    # Stage 1:
    # Essentially optim.set_parameters (re-)creates and optimizer using
    # model.paramters() as parameters that will be stored in the
    # optim.optimizer.param_groups field of the torch optimizer class.
    # Importantly, this method does not yet load the optimizer state, as
    # essentially it builds a new optimizer with empty optimizer state and
    # parameters from the model.
    optim.set_parameters(model.named_parameters())
    print(
        "Stage 1: Keys after executing optim.set_parameters" +
        "(model.parameters())")
    show_optimizer_state(optim)

    if opt.train_from:
        # Stage 2: In this stage, which is only performed when loading an
        # optimizer from a checkpoint, we load the saved_optimizer_state_dict
        # into the re-created optimizer, to set the optim.optimizer.state
        # field, which was previously empty. For this, we use the optimizer
        # state saved in the "saved_optimizer_state_dict" variable for
        # this purpose.
        # See also: https://github.com/pytorch/pytorch/issues/2830
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        # Convert back the state values to cuda type if applicable
        if use_gpu(opt):
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        print(
            "Stage 2: Keys after executing  optim.optimizer.load_state_dict" +
            "(saved_optimizer_state_dict)")
        show_optimizer_state(optim)

        # We want to make sure that indeed we have a non-empty optimizer state
        # when we loaded an existing model. This should be at least the case
        # for Adam, which saves "exp_avg" and "exp_avg_sq" state
        # (Exponential moving average of gradient and squared gradient values)
        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


# Debugging method for showing the optimizer state
def show_optimizer_state(optim):
    print("optim.optimizer.state_dict()['state'] keys: ")
    for key in optim.optimizer.state_dict()['state'].keys():
        print("optim.optimizer.state_dict()['state'] key: " + str(key))

    print("optim.optimizer.state_dict()['param_groups'] elements: ")
    for element in optim.optimizer.state_dict()['param_groups']:
        print("optim.optimizer.state_dict()['param_groups'] element: " + str(
            element))
