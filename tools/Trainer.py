from __future__ import division

import time
import sys
import math

from collections import defaultdict

import torch
import torch.nn as nn

import tools
import tools.io
import tools.modules


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, xent=0, kl=0, n_words=0, n_correct=0):
        self._xent = xent
        self._kl = kl
        self._n_words = n_words
        self._n_correct = n_correct
        self._n_src_words = 0
        self._start_time = time.time()

    def update(self, stat):
        self._xent += stat._xent
        self._kl += stat._kl
        self._n_words += stat._n_words
        self._n_correct += stat._n_correct

    def accuracy(self):
        return 100 * (self._n_correct / self._n_words)

    def xent(self):
        return self._xent / self._n_words

    def kl(self):
        return self._kl / self._n_words

    def ppl(self):
        return math.exp(min(self._xent / self._n_words, 100))

    def expelbo(self):
        return math.exp(min((self._xent + self._kl) / self._n_words, 100))

    def elapsed_time(self):
        return time.time() - self._start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; " +
               "expelbo: %6.2f; ppl: %6.2f; xent: %6.2f; kl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.expelbo(),
               self.ppl(),
               self.xent(),
               self.kl(),
               self._n_src_words / (t + 1e-5),
               self._n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

    def log_tensorboard(self, prefix, writer, lr, step):
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper",  self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", lr, step)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`tools.Model.NMTModel`): translation model to train

            train_loss(:obj:`tools.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`tools.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`tools.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1,
                 q_warmup_start=0, q_warmup_steps=0, n_attn_samples=1):

        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.progress_step = 0

        self.q_warmup_start = q_warmup_start
        self.q_warmup_steps = q_warmup_steps
        self.n_attn_samples = n_attn_samples
        self.alphas = defaultdict(lambda: 1)
        if q_warmup_steps > 0:
            for i, x in enumerate(torch.range(q_warmup_start, 1, 1 / q_warmup_steps).tolist()):
                self.alphas[i] = x

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`tools.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation(
                        true_batchs, total_stats,
                        report_stats, normalization)

                if report_func is not None:
                    if idx % 1000 == -1 % 1000:
                        print("|Param|: {}".format(sum([p.norm()**2 for p in self.model.parameters()]).item()**0.5))
                    report_stats = report_func(
                            epoch, idx, num_batches,
                            self.progress_step,
                            total_stats._start_time, self.optim.lr,
                            report_stats)
                    self.progress_step += 1
                    sys.stdout.flush()

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

        if len(true_batchs) > 0:
            self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization)
            true_batchs = []

        return total_stats

    def validate(self, valid_iter, mode=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`tools.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        old_mode = self.model.mode
        if mode is None:
            mode = old_mode
        self.model.mode = mode
        self.valid_loss.generator.mode = mode
        # self.valid_loss.generator and self.train_loss.generator are references

        stats = Statistics()

        with torch.no_grad():
            for iii,batch in enumerate(valid_iter):
                sys.stdout.flush()
                cur_dataset = valid_iter.get_cur_dataset()
                self.valid_loss.cur_dataset = cur_dataset

                src = tools.io.make_features(batch, 'src', self.data_type)
                if self.data_type == 'text':
                    _, src_lengths = batch.src
                else:
                    src_lengths = None

                tgt = tools.io.make_features(batch, 'tgt')

                # F-prop through the model.
                outputs, attns, _, dist_info, outputs_baseline = self.model(src, tgt, src_lengths)

                # Compute loss.
                batch_stats = self.valid_loss.monolithic_compute_loss(
                        batch, outputs, attns, dist_info=dist_info, output_baseline=outputs_baseline)

                # Update statistics.
                stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()
        self.model.mode = old_mode
        self.train_loss.generator.mode = old_mode

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': tools.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.expelbo(), epoch))

    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, normalization):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = tools.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats._n_src_words += src_lengths.sum()
            else:
                src_lengths = None

            tgt_outer = tools.io.make_features(batch, 'tgt')

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                outputs, attns, dec_state, dist_info, outputs_baseline = \
                    self.model(src, tgt, src_lengths, dec_state)

                # 3. Compute loss in shards for memory efficiency.
                self.train_loss.alpha = self.alphas[self.progress_step]
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization,
                        dist_info=dist_info, output_baseline=outputs_baseline)

                # nan-check
                nans = [
                    (name, param)
                    for name, param in self.model.named_parameters()
                    if param.grad is not None and (param.grad != param.grad).any()
                ]
                if nans:
                    print("FOUND NANS")
                    print([x[0] for x in nans])
                    for _, param in nans:
                        param.grad[param.grad!=param.grad] = 0

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        if self.grad_accum_count > 1:
            self.optim.step()
