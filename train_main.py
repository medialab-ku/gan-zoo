import argparse
import importlib
import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from core.log import init_logger
from core.saver import Saver

cudnn.benchmark = True

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--config',
                        help='configure file name of the experiment',
                        default='dcgan',
                        type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    user_config = importlib.import_module('user_config.' + args.config.replace('/', '.'))
    gen_config = user_config.GenConfig()
    disc_config = user_config.DiscConfig()

    L = init_logger(gen_config.tag, gen_config.ckpt_path)
    L.info("set config G: %s" % gen_config)
    L.info("set config D: %s" % disc_config)

    gen_saver = Saver(gen_config)
    generator, gen_optim, global_step, last_epoch = gen_saver.load()

    disc_saver = Saver(disc_config)
    discriminator, disc_optim, _, _ = disc_saver.load()

    models = {
        'gen': generator,
        'disc': discriminator
    }

    optims = {
        'gen': gen_optim,
        'disc': disc_optim
    }

    gen_to_save = generator
    disc_to_save = discriminator

    if gen_config.multi_gpu:
        models['gen'] = torch.nn.DataParallel(generator)
        models['disc'] = torch.nn.DataParallel(discriminator)

    gen_scheduler, disc_scheduler = None, None
    if hasattr(gen_config, 'scheduler'):
        gen_config.scheduler_param['last_epoch'] = -1
        gen_scheduler = gen_config.scheduler(gen_optim, **gen_config.scheduler_param)
        disc_scheduler = gen_config.scheduler(disc_optim, **gen_config.scheduler_param)

    log_dir = os.path.join(gen_config.ckpt_path, 'tb')
    writer = SummaryWriter(log_dir=log_dir)
    trainer = gen_config.trainer(gen_config, models, optims, writer)
    # validator = gen_config.validator(gen_config, generator, writer)

    for epoch in range(last_epoch + 1, gen_config.epoch):
        _, global_step, avg_loss = trainer.step(epoch, global_step)
        L.info('Training epoch %d was done. (avg_loss: %f)' % (epoch, avg_loss))

        # result, avg_acc = validator.step(epoch, global_step)
        # L.info('Validation epoch %d was done. (avg_acc: %f)' % (epoch, avg_acc))

        L.info('Saving the trained generator model... (%d epoch, %d step)' % (epoch, global_step))
        gen_saver.save(gen_to_save, gen_optim, global_step, epoch,
                       performance=0,
                       perf_op='lt')
        L.info('Saving G is finished.')

        L.info('Saving the trained discriminator model... (%d epoch, %d step)' % (epoch, global_step))
        disc_saver.save(disc_to_save, disc_optim, global_step, epoch,
                        performance=0,
                        perf_op='lt')
        L.info('Saving D is finished.')

        if gen_scheduler:
            gen_scheduler.step(epoch)
            disc_scheduler.step(epoch)


if __name__ == '__main__':
    main()
