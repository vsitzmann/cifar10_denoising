import argparse
import os, datetime
import torch

from dataio import NoisyCIFAR10Dataset
from torch.utils.data import DataLoader
from denoising_unet import DenoisingUnet
from tensorboardX import SummaryWriter

# params
parser = argparse.ArgumentParser()

# data paths
parser.add_argument('--data_root', required=True, help='path to file list of h5 train data')
parser.add_argument('--logging_root', type=str, default='/media/staging/deep_sfm/',
                    required=False, help='path to file list of h5 train data')

# train params
parser.add_argument('--train_test', type=str, required=True, help='path to file list of h5 train data')
parser.add_argument('--experiment_name', type=str, default='', help='path to file list of h5 train data')
parser.add_argument('--checkpoint', type=str, default=None, help='path to file list of h5 train data')
parser.add_argument('--max_epoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--sigma', type=float, default=0.05, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.001')
parser.add_argument('--batch_size', type=int, default=4, help='start epoch')

parser.add_argument('--reg_weight', type=int, default=0., help='start epoch')

opt = parser.parse_args()
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

device = torch.device('cuda')


def params_to_filename(params):
    params_to_skip = ['batch_size', 'max_epoch', 'train_test']
    fname = ''
    for key, value in vars(params).items():
        if key in params_to_skip:
            continue
        if key == 'checkpoint' or key == 'data_root' or key == 'logging_root':
            if value is not None:
                value = os.path.basename(os.path.normpath(value))

        fname += "%s_%s_" % (key, value)
    return fname


def train(model, dataset):
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size)

    if opt.checkpoint is not None:
        model.load_state_dict(torch.load(opt.checkpoint))

    model.train()
    model.cuda()

    # directory structure: month_day/
    dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),
                            datetime.datetime.now().strftime('%H-%M-%S_')) + params_to_filename(opt)

    log_dir = os.path.join(opt.logging_root, 'logs', dir_name)
    run_dir = os.path.join(opt.logging_root, 'runs', dir_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    writer = SummaryWriter(run_dir)
    iter = 0

    writer.add_scalar("learning_rate", opt.lr, 0)

    print('Beginning training...')
    for epoch in range(opt.max_epoch):
        for model_input, ground_truth in dataloader:
            ground_truth = ground_truth.cuda()
            model_input = model_input.cuda()

            model_outputs = model(model_input)
            model.write_updates(writer, model_outputs, ground_truth, model_input, iter)

            optimizer.zero_grad()

            dist_loss = model.get_distortion_loss(model_outputs, ground_truth)
            reg_loss = model.get_regularization_loss(model_outputs, ground_truth)

            total_loss = dist_loss + opt.reg_weight * reg_loss

            total_loss.backward()
            optimizer.step()

            print("Iter %07d   Epoch %03d   dist_loss %0.4f reg_loss %0.4f" %
                  (iter, epoch, dist_loss, reg_loss * opt.reg_weight))

            writer.add_scalar("scaled_regularization_loss", reg_loss * opt.reg_weight, iter)
            writer.add_scalar("distortion_loss", dist_loss, iter)

            if not iter:
                # Save parameters used into the log directory.
                with open(os.path.join(log_dir, "params.txt"), "w") as out_file:
                    out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

            iter += 1
            if iter % 10000 == 0:
                torch.save(model.state_dict(), os.path.join(log_dir, 'model-epoch_%d_iter_%s.pth' % (epoch, iter)))

    torch.save(model.state_dict(), os.path.join(log_dir, 'model-epoch_%d_iter_%s.pth' % (epoch, iter)))


def main():
    dataset = NoisyCIFAR10Dataset(data_root=opt.data_root,
                                 sigma=opt.sigma,
                                 train=opt.train_test == 'train')
    model = DenoisingUnet(img_sidelength=32)
    train(model, dataset)


if __name__ == '__main__':
    main()
