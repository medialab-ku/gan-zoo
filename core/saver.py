import operator
import os

import torch
from parse import parse

from core.tags import *

__all__ = ['Saver']


class Saver(object):
    def __init__(self, config, save_range=50):
        self.config = config
        self.save_range = save_range
        self.param_dir = os.path.join(self.config.ckpt_path, 'param')
        if not os.path.exists(self.param_dir):
            os.makedirs(self.param_dir)

        self.params = self.get_params_on_path(self.config.max_to_keep)
        self.best_perf, self.best_list = self.get_best_perf()

    def get_best_perf(self):
        perf = None

        best_epochs = list()
        if os.path.exists(self.param_dir):
            file_list = os.listdir(self.param_dir)
            for param in file_list:
                parsed_list = parse(BEST_CKPT, param)
                if parsed_list:
                    best_epoch = parsed_list[0]
                    best_epochs.append(best_epoch)

        best_epochs = sorted(best_epochs)

        # best_ckpt = os.path.join(self.param_dir, BEST
        if len(best_epochs) > 0:
            ckpt = torch.load(os.path.join(self.param_dir, BEST_CKPT.format(best_epochs[-1])))
            perf = ckpt[PERFORMANCE]

        return perf, best_epochs

    def get_params_on_path(self, max_to_keep):
        params = []
        if os.path.exists(self.param_dir):
            file_list = os.listdir(self.param_dir)
            for param in file_list:
                parsed_list = parse(CKPT_FMT, param)
                if parsed_list:
                    epoch = parsed_list[0]
                    params.append(epoch)

                    params = sorted(params)
                    params = params[-max_to_keep:]

        return params

    def load(self):
        model, optim, step, epoch = self.create()

        if self.params:
            ckpt = torch.load(os.path.join(self.param_dir, CKPT_FMT.format(self.params[-1])),
                              map_location=self.config.device)
            model.load_state_dict(ckpt[STATE])
            model.to(self.config.device)
            optim.load_state_dict(ckpt[OPTIM])  # may create new optimizer

            step = ckpt[STEP]
            epoch = ckpt[EPOCH]
            print("Loaded the model at {} epoch and {} step".format(epoch, step))
        else:
            print("Start from scratch ... ")

        return model, optim, step, epoch

    def load_target(self, target_epoch):
        model, optim, step, epoch = self.create()

        for dirpath, dirnames, filenames in os.walk(self.param_dir):
            for filename in filenames:
                parsed_list = parse(CKPT_FMT, filename)
                if parsed_list is not None:
                    epoch = parsed_list[-1]
                    if epoch == target_epoch:
                        ckpt = torch.load(
                            os.path.join(self.param_dir, filename))
                        model.load_state_dict(ckpt[STATE])
                        optim.load_state_dict(ckpt[OPTIM])
                        step = ckpt[STEP]
                        epoch = ckpt[EPOCH]

                        return model, optim, step, epoch

        raise Exception('Cannot find specific model: epoch %d' % target_epoch)

    def save(self, model, optim, step, epoch, performance=None, perf_op='gt'):
        if perf_op == 'gt':
            _op = operator.gt
        elif perf_op == 'lt':
            _op = operator.lt
        else:
            raise ValueError("perf_op ({})is not supported".format(perf_op))

        filename_to_save = os.path.join(self.param_dir, CKPT_FMT.format(epoch))

        torch.save(
            {
                STATE: model.state_dict(),
                OPTIM: optim.state_dict(),
                STEP: step,
                EPOCH: epoch,
                PERFORMANCE: performance
            },
            filename_to_save
        )
        self.params.append(epoch)

        if len(self.params) > self.config.max_to_keep:
            param_to_del = self.params.pop(0)
            filepath_to_del = os.path.join(self.param_dir,
                                           CKPT_FMT.format(param_to_del))
            os.remove(filepath_to_del)

        if performance:
            quot = epoch // self.save_range
            lquot = 0
            if len(self.best_list) > 0:
                latest_best_epoch = self.best_list[-1]
                lquot = latest_best_epoch // self.save_range

            if quot > lquot:
                # save new perf
                filename_to_save = os.path.join(self.param_dir, BEST_CKPT.format(epoch))
                torch.save(
                    {
                        STATE: model.state_dict(),
                        OPTIM: optim.state_dict(),
                        STEP: step,
                        EPOCH: epoch,
                        PERFORMANCE: performance
                    },
                    filename_to_save
                )
                self.best_perf = performance
                self.best_list.append(epoch)

            if not self.best_perf or _op(performance, self.best_perf):
                if len(self.best_list) > 0:
                    param_to_del = self.best_list.pop()  # get last best perf
                    filepath_to_del = os.path.join(self.param_dir, BEST_CKPT.format(param_to_del))
                    os.remove(filepath_to_del)  # remove

                torch.save(
                    {
                        STATE: model.state_dict(),
                        OPTIM: optim.state_dict(),
                        STEP: step,
                        EPOCH: epoch,
                        PERFORMANCE: performance
                    },
                    os.path.join(self.param_dir, BEST_CKPT.format(epoch))
                )
                self.best_perf = performance
                self.best_list.append(epoch)

    def create(self):
        model = self.config.model(**self.config.model_param).to(self.config.device)
        if hasattr(model, 'init_weights'):
            model.init_weights()

        if hasattr(model, 'transfer_weights') and hasattr(self.config, 'transfer_param'):
            model.transfer_weights(**self.config.transfer_param)

        optim = self.config.optim(model.parameters(), **self.config.optim_param)
        step = 0
        epoch = -1
        return model, optim, step, epoch
