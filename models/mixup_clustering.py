from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug
import numpy as np
import torch

class MixupClustering(ContinualModel):
    NAME = 'mixupKM'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_lodaer, transform):
        super(MixupClustering, self).__init__(backbone, loss, args, len_train_lodaer, transform)
        # self.buffer = Buffer(self.args.model.buffer_size, self.device)
        # self.buffer = Buffer(self.args.model.buffer_size, self.device, n_tasks=self.args.n_tasks, mode='ring')

    def observe(self, inputs1, labels, inputs2, notaug_inputs, buffer, epoch):

        self.opt.zero_grad()
        if buffer.is_empty():
            if self.args.cl_default:
                labels = labels.to(self.device)
                outputs = self.net.module.backbone(inputs1.to(self.device))
                loss = self.loss(outputs, labels).mean()
                data_dict = {'loss': loss}

            else:
                data_dict = self.net.forward(inputs1.to(self.device, non_blocking=True),
                                             inputs2.to(self.device, non_blocking=True),
                                             notaug_inputs.to(self.device, non_blocking=True),
                                             epoch)

                if 'loss' in data_dict.keys():
                    loss = data_dict['loss'].mean()
                    data_dict['loss'] = data_dict['loss'].mean()
                else:
                    siam_loss = data_dict['siam_L'].mean()
                    recon_loss = data_dict['recon_L'].mean()
                    data_dict['siam_L'] = data_dict['siam_L'].mean()
                    data_dict['recon_L'] = data_dict['recon_L'].mean()
                data_dict['penalty'] = 0.0
        else:
            if self.args.cl_default:
                buf_inputs, buf_labels = buffer.get_data(self.args.train.batch_size, transform=self.transform)
                buf_labels = buf_labels.to(self.device).long()
                labels = labels.to(self.device).long()
                lam = np.random.beta(self.args.train.alpha, self.args.train.alpha)
                mixed_x = lam * inputs1.to(self.device) + (1 - lam) * buf_inputs[:inputs1.shape[0]].to(self.device)
                net_output = self.net.module.backbone(mixed_x.to(self.device, non_blocking=True))
                buf_labels = buf_labels[:inputs1.shape[0]].to(self.device)
                loss = self.loss(net_output, labels) + (1 - lam) * self.loss(net_output, buf_labels)
                data_dict = {'loss': loss}
                data_dict['penalty'] = 0.0
            else:
                if buffer.buffer_size == 0:
                    mixed_x = inputs1.to(self.device)
                    mixed_x_aug = inputs2.to(self.device)
                else:
                    buf_inputs1, buf_inputs2 = buffer.get_data(self.args.train.batch_size, transform=self.transform)
                    lam = np.random.beta(self.args.train.alpha, self.args.train.alpha)
                    mixed_x = lam * inputs1.to(self.device) + (1 - lam) * buf_inputs1[:inputs1.shape[0]].to(self.device)
                    mixed_x_aug = lam * inputs2.to(self.device) + (1 - lam) * buf_inputs2[:inputs1.shape[0]].to(self.device)
                data_dict = self.net.forward(mixed_x.to(self.device, non_blocking=True),
                                             mixed_x_aug.to(self.device, non_blocking=True),
                                             notaug_inputs.to(self.device, non_blocking=True),
                                             epoch)

                if 'loss' in data_dict.keys():
                    loss = data_dict['loss'].mean()
                    data_dict['loss'] = data_dict['loss'].mean()
                else:
                    siam_loss = data_dict['siam_L'].mean()
                    recon_loss = data_dict['recon_L'].mean()
                    data_dict['siam_L'] = data_dict['siam_L'].mean()
                    data_dict['recon_L'] = data_dict['recon_L'].mean()
                data_dict['penalty'] = 0.0

        if 'loss' in data_dict.keys():
            loss.backward()
        else:
            recon_loss.backward()
            siam_loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.args.train.base_lr})
        if self.args.cl_default:
            buffer.add_data(examples=notaug_inputs, logits=labels)
        else:
            if not hasattr(buffer, 'examples'):
                buffer.add_data(examples=notaug_inputs, logits=inputs1)

        return data_dict, buffer
