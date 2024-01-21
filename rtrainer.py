import sys
sys.path.append('../')

from utils.util import check_parameters
import time
import logging
from logger.set_logger import setup_logger
from model.loss import Loss, mlloss
import torch
import os
from torch.nn.parallel import data_parallel

class Trainer(object):
    def __init__(self, train_dataloader, val_dataloader, spec_model,  optimizer, scheduler, opt):
        super(Trainer).__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.scheduler = scheduler
        self.cur_epoch = 0
        self.total_epoch = opt.epochs
        self.early_stop = opt.early_stop

        self.print_freq = opt.print_freq
        self.logger = logging.getLogger(opt.logger_name)
        self.save_path = opt.save_folder
        
        self.sinc = opt.sinc
        self.weights = opt.weights
        self.resume_state = opt.resume_state
        self.episod = opt.episod
        if opt.train_gpuid:
            self.logger.info('Load Nvida GPU .....')
            self.logger.info(torch.cuda.device_count())
            self.device = torch.device(
                'cuda:{}'.format(opt.train_gpuid[0]))
            self.gpuid = opt.train_gpuid
            if opt.resume_state == 0:
                self.dualrnn = spec_model.to(self.device)
                self.logger.info(
                'Loading Model parameters: {:.3f} Mb'.format(check_parameters(self.dualrnn)))
        else:
            self.logger.info('Load CPU ...........')
            self.device = torch.device('cpu')
            self.dualrnn = spec_model.to(self.device)
            self.logger.info(
                'Loading Model parameters: {:.3f} Mb'.format(check_parameters(self.dualrnn)))

        if opt.resume_state:
            ckp = torch.load(opt.resume_path, map_location='cpu')
            self.cur_epoch = ckp['epoch']
            self.logger.info("Resume from checkpoint {}: epoch {:.3f}".format(
                opt.resume_path, self.cur_epoch))
           
            if opt.resume_state == 2:
                model_dict = spec_model.state_dict()
                pretrained_dict = ckp['model_state_dict']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                spec_model.load_state_dict(model_dict)
            else:
                spec_model.load_state_dict(
                    ckp['model_state_dict'])

            self.dualrnn = spec_model.to(self.device)
            self.logger.info(
                 'Loading Model parameters: {:.3f} Mb'.format(check_parameters(self.dualrnn)))
       #     optimizer.load_state_dict(ckp['optim_state_dict'])
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer
        
        if opt.max_norm:
            self.clip_norm = opt.max_norm
            self.logger.info(
                "Gradient clipping by {}, default L2".format(self.clip_norm))
        else:
            self.clip_norm = 0

    def train(self, epoch):
        self.logger.info(
            'Start training from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.dualrnn.train()
        num_batchs = len(self.train_dataloader)
        total_loss = 0.0
        total_loss_hr = 0.0
        total_loss_snr = 0.0
        start_time = time.time()
        for iterno, load_data in enumerate(self.train_dataloader):
            data_lr = load_data[0].to(self.device)
            data_orig = load_data[1].to(self.device)
            total_outs = []
            total_outputs = []
            total_rewards = []
            total_action_prob = []
            if self.gpuid:
                inputs = data_lr
                first_inputs = data_lr
                for i in range(self.episod):
                    outs, outputs, act_prob = self.dualrnn(inputs)
                    total_outs.append(outs)
                    total_outputs.append(outputs)
                    out_reward = mlloss(inputs, outputs.detach(), data_orig)
                    total_rewards.append(out_reward)
                    total_action_prob.append(act_prob)
                    inputs = outputs.detach()
            for i in range(self.episod):
                self.optimizer.zero_grad()
                G = total_rewards[i]
                alpha = 0.1 
                for j in range(i+1, self.episod):
                    G = G + alpha * total_rewards[j]
                    alpha = alpha * alpha
                action_prob = 0.0001 * torch.sum(total_action_prob[i], dim=1)
                state_value = -1.0 * torch.mean(G * action_prob)
                epoch_loss, epoch_loss_snr = Loss(total_outs[i], total_outputs[i], data_orig) 
                (state_value + self.weights * epoch_loss).backward()
                 
                if self.clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.dualrnn.parameters(), self.clip_norm)
                  
                self.optimizer.step()
 
            epoch_loss, epoch_loss_snr = Loss(inputs, data_orig)
            epoch_loss_hr = epoch_loss


            total_loss_hr += epoch_loss_hr.item()
            total_loss += epoch_loss.item()
            total_loss_snr += epoch_loss_snr.item()

            if (iterno+1) % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, loss_hr:{:.3f}, snr:{:.3f}>'.format(
                    epoch, iterno+1, self.optimizer.param_groups[0]['lr'], total_loss/(iterno+1), total_loss_hr/(iterno+1), total_loss_snr/(iterno+1))
                self.logger.info(message)
        end_time = time.time()
        total_loss = total_loss/(iterno + 1)
        total_loss_snr = total_loss_snr/(iterno + 1)
        total_loss_hr = total_loss_hr/(iterno + 1)
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, loss_hr:{:.3f}, snr:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, iterno+1, self.optimizer.param_groups[0]['lr'], total_loss, total_loss_hr, total_loss_snr, (end_time-start_time)/60)
        self.logger.info(message)
        return total_loss, total_loss_snr

    def validation(self, epoch):
        self.logger.info(
            'Start Validation from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.dualrnn.eval()
        num_batchs = len(self.val_dataloader)
        total_loss = 0.0
        total_loss_snr = 0.0
        total_loss_hr = 0.0
        start_time = time.time()
        with torch.no_grad():
            for iterno, load_data in enumerate(self.val_dataloader):
                data_lr = load_data[0].to(self.device)
                data_orig = load_data[1].to(self.device)

                if self.gpuid:
                    inputs = data_lr
                    for i in range(self.episod):
                        outs, outputs, out_prob = self.dualrnn(inputs)
                        inputs = outputs.detach()
                
                epoch_loss, epoch_loss_snr = Loss(outs, inputs, data_orig)
                epoch_loss_hr = epoch_loss

                total_loss_hr += epoch_loss_hr.item()  
                total_loss += epoch_loss.item()
                total_loss_snr += epoch_loss_snr.item()

                if (iterno+1) % self.print_freq == 0:
                    message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, loss_hr:{:.3f}, snr:{:.3f}>'.format(
                        epoch, iterno+1, self.optimizer.param_groups[0]['lr'], total_loss/(iterno+1), total_loss_hr/(iterno+1), total_loss_snr/(iterno+1))
                    self.logger.info(message)
 
        end_time = time.time() 
        total_loss = total_loss/(iterno+1)
        total_loss_snr = total_loss_snr/(iterno+1)
        total_loss_hr = total_loss_hr/(iterno + 1)
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, loss_hr:{:.3f}, snr:{:.3f}, Total time:{:.3f} min> '.format(epoch, iterno+1, self.optimizer.param_groups[0]['lr'], total_loss, total_loss_hr, total_loss_snr, (end_time-start_time)/60)
        self.logger.info(message)
        return total_loss, total_loss_snr

    def run(self):
        train_loss = []
        val_loss = []
#        while 1:
        with torch.cuda.device(self.gpuid[0]):
            v_loss, v_snr = self.validation(self.cur_epoch)
            best_loss = v_loss

            if self.resume_state:
                best_loss = 1000000.0

            self.logger.info("Starting epoch from {:d}, loss = {:.4f}".format(
                self.cur_epoch, best_loss))
            no_improve = 0
            # starting training part
            while self.cur_epoch < self.total_epoch:
                self.cur_epoch += 1
                t_loss, t_snr = self.train(self.cur_epoch)
                v_loss, v_snr = self.validation(self.cur_epoch)

                train_loss.append(t_loss)
                val_loss.append(v_loss)

                # schedule here
                self.scheduler.step(v_loss)

                if v_loss >= best_loss:
                    no_improve += 1
                    self.logger.info(
                        'No improvement, Best Loss: {:.4f}'.format(best_loss))
                else:
                    best_loss = v_loss
                    no_improve = 0
                    self.save_checkpoint(self.cur_epoch, t_loss, v_loss, t_snr, v_snr)
                    self.logger.info('Epoch: {:d}, Now Best Loss Change: {:.4f}'.format(
                        self.cur_epoch, best_loss))

                if no_improve == self.early_stop:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_improve))
                    break
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, self.total_epoch))


    def save_checkpoint(self, epoch, t_loss, v_loss, t_snr, v_snr):
        '''
           save model
           best: the best model
        '''
        os.makedirs(self.save_path, exist_ok=True)
        ckpt_name = 'nnet_iter%d_trloss%.4f_trsnr%.4f_valoss%.4f_vasnr%.4f_epoch%d.pt' %(
                    epoch, t_loss, t_snr, v_loss, v_snr, epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.dualrnn.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        },
            os.path.join(self.save_path, ckpt_name))
