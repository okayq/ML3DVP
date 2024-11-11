import torch
device = torch.device('cuda')
import os
import os.path as osp
import json
import torch
import torch.nn.functional as F
import pickle
import logging
import numpy as np
from model import SimVP
from tqdm import tqdm
from API import *
from utils import * 
from skimage.metrics import structural_similarity as cal_ssim
from PIL import Image
import cv2

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()
        self._preparation()
        print_log(output_namespace(self.args))
        self._get_data()
        self._select_optimizer()
        self._select_criterion()

    def _acquire_device(self):
        device = torch.device('cuda')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader = load_data(**config)
        # print('34')
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader
        # print('45')

    # start
    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    # loss calculation
    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()
        
    def psnr_loss(self, img1, img2):
        mse_loss = torch.mean((img1 * 255 - img2 * 255)**2)
        psnr = 20 * torch.log10(torch.tensor(255.0)) - 10 * torch.log10(mse_loss)
        return psnr
    
    def calculate_loss(self, outputs, targets):
        loss_psnr = 0.0
        loss_ssim = 0.0
        
        for b in range(outputs.shape[0]):
            for f in range(outputs.shape[1]):
                output = outputs[b,f]
                target = targets[b,f]
                # print(output.shape, target.shape)
                output_np = output.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                psnr = self.psnr_loss(output, target)
                ssim = cal_ssim(output_np.swapaxes(0, 2), target_np.swapaxes(0,2), multichannel=True, channel_axis=2, data_range=1)
                loss_psnr += psnr.item()
                loss_ssim += ssim.item()
                
        loss_psnr /= (outputs.shape[0] * outputs.shape[1])
        loss_ssim /= (outputs.shape[0] * outputs.shape[1])
        
        return loss_psnr, loss_ssim
    # end

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)
        if torch.cuda.device_count() > 1:
            print("let's use", torch.cuda.device_count(), "gpus")
            self.model = torch.nn.DataParallel(self.model)
        print('1')
        self.model.to(self.device)
        print('2')
        print('device', self.device)
        
        # lambda_reg = 0.01
        current_path = os.getcwd()
        checkpoint_path = os.path.join(current_path, 'data/human/checkpoint.pth')  # 替换为您的权重文件路径
        checkpoint = torch.load(checkpoint_path)
        if torch.cuda.device_count() > 1:
            # 多GPU训练模型，需要添加"module."前缀
            new_state_dict = {}
            for key, value in checkpoint.items():
                # new_key = 'module.' + key
                new_state_dict[key] = value
            self.model.load_state_dict(new_state_dict)
            print('ok')
        else:
            # 单GPU训练模型，直接加载权重
            self.model.load_state_dict(state_dict)
            
#         fisher_estimates = {name:torch.zeros_like(param) for name, param in self.model.module.named_parameters() if 'weight' in name}
#         param_prev = {name:None for name, _ in self.model.module.named_parameters()}
    
        # start
        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            l2_regularization = torch.tensor(0.).to(self.device)
            train_pbar = tqdm(self.train_loader)

            for batch_x, batch_y in train_pbar:
                epoch = epoch
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device, non_blocking=True)
                pred_y = self.model(batch_x)
                # print(pred_y.shape, batch_y.shape)
                loss_mse = self.criterion(pred_y, batch_y)
                loss_psnr, loss_ssim = self.calculate_loss(pred_y, batch_y)
                
                # ewc_loss = 0.0
                # for name, param in self.model.module.named_parameters():
                #     if 'weight' in name:
                #         fisher_estimate = fisher_estimates.get(name.replace('module.', ''), torch.zeros_like(param))
                #         if param.grad is not None:
                #             if param_prev[name] is not None:
                #                 ewc_loss += (fisher_estimate * torch.square(param - param_prev[name])).sum()
                #             fisher_estimates[name] = fisher_estimates[name] + torch.square(param.grad)
                #         param_prev[name] = param.detach().clone()
                        
#                 l2_regularization = 0.0
#                 for name, param in self.model.named_parameters():
#                     if 'weight' in name:
#                         l2_regularization += torch.norm(param, p=2).to(self.device)
 
                
                # loss = 0.9 * loss_mse + 0.0001 * loss_psnr + 0.001 * loss_ssim  +  0.01 * lambda_reg * l2_regularization
                # else:
                loss = 0.9 * loss_mse + 0.0001 * loss_psnr + 0.001 * loss_ssim
                    
                # print(ewc_loss)
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_loss = np.average(train_loss)
            # end
            
            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_mse_loss = self.vali(self.vali_loader, epoch)
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch))
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
                    epoch + 1, train_loss, vali_mse_loss))
                recorder(vali_mse_loss, self.model, self.path)
        # start
        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model
        # end
    def vali(self, vali_loader, m):
        epoch = m
        # lambda_reg = 0.01
        self.model.eval()
        preds_lst, trues_lst, mse_loss = [], [], []
        # l2_regularization = torch.tensor(0.).to(self.device)
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            loss_mse = self.criterion(pred_y, batch_y)
         
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss_mse.mean().item()))
            # total_loss.append(loss.mean().item())
            mse_loss.append(loss_mse.mean().item())

        # total_loss = np.average(total_loss)
        mse_loss = np.average(mse_loss)
        # start
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        mse, mae, loss_ssim, loss_psnr = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, loss_ssim, loss_psnr))
        # end
        self.model.train()
        return mse_loss

    def test(self, args):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        for batch_x, batch_y in self.test_loader:
            pred_y = self.model(batch_x.to(self.device))
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        print(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # mse, mae = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std)
        # psnr, ssim = self.calculate_loss(preds, trues)
        mse, mae, ssim, psnr = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, True)
        
        print_log('mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
    
            
        return mse