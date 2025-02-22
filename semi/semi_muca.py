import os

import torch
from nets.segformer_training import (CE_Loss, Dice_loss, Focal_Loss,DiceLoss,softmax_mse_loss,
                                     weights_init)
from tqdm import tqdm

from utils.utils import get_lr

from utils.utils_metrics import f_score
import numpy as np
from torch.nn import functional as F
from PIL import Image
import random
from nets.ctsa import CTSA
from torch.nn import Softmax,LayerNorm
from torch import  nn
import torchvision.transforms as transforms
from utils.dataloader_unlabel import cutmix_images,SA,WA

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    return 3 * sigmoid_rampup(epoch, 150)

def multi_head_rep(x):
    new_x_shape = x.size()[:-1] + (3, 256)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

def attn_norm(x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    map_in = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, padding=0),
                           nn.GELU())
    attn_norm = LayerNorm(256, eps=1e-6)
    map_in.to(device)
    attn_norm.to(device)
    x = map_in(x)
    x = x.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)
    x = attn_norm(x)
    return x

def fit_one_epoch(model_train, model,model_train_unlabel,ema_model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,gen, gen_unlabel,gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
    fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    suloss_item=0
    consistency_loss_item=0
    attn_loss_item=0
    loss_u=0
    val_loss        = 0
    val_loss_at =0
    val_f_score     = 0
    dropout = nn.Dropout(p=0.5)
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    dice_loss = DiceLoss(num_classes)
    criterion_u = nn.CrossEntropyLoss().cuda(local_rank)
    for iteration, ((imgs_label, pngs, labels),imgs_unlabel) in enumerate(zip(gen,gen_unlabel)):
        if iteration >= epoch_step: 
            break
        imgs_unlabel = imgs_unlabel
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs_unlabel_w= WA(imgs_unlabel)
                imgs_unlabel_s = SA(imgs_unlabel, imgs_label)
                imgs_label = imgs_label.cuda(local_rank)
                imgs_unlabel    = imgs_unlabel.cuda(local_rank)
                imgs_unlabel_w = imgs_unlabel_w.cuda(local_rank)
                imgs_unlabel_s = imgs_unlabel_s.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

                model_train.eval()
                _, _, _,_, pred_u_pseudo,_ = model_train(imgs_unlabel)
                pred_u_pseudo = pred_u_pseudo.detach()
                model_train.module.set_pseudo_prob_map(pred_u_pseudo)
                pseudo_label = pred_u_pseudo.argmax(dim=1)
                model_train.module.set_pseudo_label(pseudo_label)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if fp16:
            # ----------------------#
            #
            # ----------------------#
            model_train.train()
            # model_train_unlabel.train()
            num_lb, num_ulb = imgs_label.shape[0], imgs_unlabel_s.shape[0]
            low_level_features,the_two_features,the_three_features,the_four_features,outputs_total,s_encoder = model_train(torch.cat((imgs_label, imgs_unlabel_s)))
            outputs_label, outputs_unlabel = outputs_total.split([num_lb, num_ulb])
            # s_encoder_label,s_encoder_unlabel=s_encoder.split([num_lb, num_ulb])
            _,low_level_features=low_level_features.split([num_lb, num_ulb])
            _, the_two_features = the_two_features.split([num_lb, num_ulb])
            _,the_three_features=the_three_features.split([num_lb, num_ulb])
            _,the_four_features=the_four_features.split([num_lb, num_ulb])
            #----------------------#
            #   监督学习
            #----------------------#
            if focal_loss:
                suloss = Focal_Loss(outputs_label, pngs, weights, num_classes = num_classes)
            else:
                suloss = CE_Loss(outputs_label, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs_label, labels)
                suloss      = suloss + main_dice
            # ----------------------#
            #   一致性正则化
            # ----------------------#
            noise = torch.clamp(torch.randn_like(imgs_unlabel_w) * 0.1, -0.2, 0.2)
            ema_inputs = imgs_unlabel_w + noise
            with torch.no_grad():
                low_level_features_,the_two_features_,the_three_features_,the_four_features_,ema_output,t_encoder = ema_model(ema_inputs)
            consistency_weight = get_current_consistency_weight(epoch // 2)

            T = 4
            _, _, w, h = imgs_unlabel_w.shape
            volume_batch_r = imgs_unlabel_w.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, num_classes, w, h]).cuda()
            for i in range(T // 2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                ema_inputs = dropout(ema_inputs)
                with torch.no_grad():
                    low_level_features_preds_,the_two_features_preds_,the_three_features_preds_,the_four_features__preds_,preds[2 * stride * i:2 * stride * (i + 1)],_  = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, num_classes, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * \
                          torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)

            consistency_dist = F.huber_loss(
                outputs_unlabel, ema_output, delta=1.0)
            threshold = (0.5 + 0.5 * sigmoid_rampup(epoch, 100)) * np.log(2)
            mask = (uncertainty < threshold).float()

            # consistency_standard和consistency_loss_out可以选择使用，一个是标准L_C，一个是计算了不确定性
            consistency_loss_out = torch.sum(
                mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
            # 和consistency_loss_out选择一个使用，也可以同时使用
            consistency_standard=consistency_dist

            ###################################多层次不确定性MSUC##############################################
            T = 4
            _, c, w, h = low_level_features_preds_.shape
            volume_batch_r = low_level_features_preds_.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            low_level_features_preds = torch.zeros([stride * T, c, w, h]).cuda()
            for i in range(T // 2):
                low_level_features_preds[2*stride * i:2*stride * (i + 1)]=volume_batch_r
            low_level_features_preds = F.softmax(low_level_features_preds, dim=1)
            low_level_features_preds = low_level_features_preds.reshape(T, stride, c, w, h)
            low_level_features_preds = torch.mean(low_level_features_preds, dim=0)
            uncertainty = -1.0 * \
                          torch.sum(low_level_features_preds * torch.log(low_level_features_preds + 1e-6), dim=1, keepdim=True)
            low_level_consistency_dist = F.huber_loss(
                low_level_features, low_level_features_, delta=1.0)
            threshold = (0.5 + 0.5 * sigmoid_rampup(epoch, 100)) * np.log(2)
            mask = (uncertainty < threshold).float()
            low_level_features_consistency_loss = torch.sum(
                mask * low_level_consistency_dist) / (2 * torch.sum(mask) + 1e-16)

            T = 4
            _, c, w, h = the_two_features_preds_.shape
            volume_batch_r = the_two_features_preds_.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            the_two_features_preds = torch.zeros([stride * T, c, w, h]).cuda()
            for i in range(T // 2):
                the_two_features_preds[2 * stride * i:2 * stride * (i + 1)] = volume_batch_r
            the_two_features_preds = F.softmax(the_two_features_preds, dim=1)
            the_two_features_preds = the_two_features_preds.reshape(T, stride, c, w, h)
            the_two_features_preds = torch.mean(the_two_features_preds, dim=0)
            uncertainty = -1.0 * \
                          torch.sum(the_two_features_preds * torch.log(the_two_features_preds + 1e-6), dim=1,
                                    keepdim=True)
            the_two_consistency_dist = F.huber_loss(
                the_two_features, the_two_features_, delta=1.0)
            threshold = (0.5 + 0.5 * sigmoid_rampup(epoch, 100)) * np.log(2)
            mask = (uncertainty < threshold).float()
            the_two_features_consistency_loss = torch.sum(
                mask * the_two_consistency_dist) / (2 * torch.sum(mask) + 1e-16)

            T = 4
            _, c, w, h = the_three_features_preds_.shape
            volume_batch_r = the_three_features_preds_.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            the_three_features_preds = torch.zeros([stride * T, c,w, h]).cuda()
            for i in range(T // 2):
                the_three_features_preds[2*stride * i:2*stride * (i + 1)] = volume_batch_r
            the_three_features_preds = F.softmax(the_three_features_preds, dim=1)
            the_three_features_preds = the_three_features_preds.reshape(T, stride, c, w, h)
            the_three_features_preds = torch.mean(the_three_features_preds, dim=0)
            uncertainty = -1.0 * \
                          torch.sum(the_three_features_preds * torch.log(the_three_features_preds + 1e-6), dim=1,
                                    keepdim=True)
            the_three_consistency_dist = F.huber_loss(
                the_three_features, the_three_features_, delta=1.0)
            threshold = (0.5 + 0.5 * sigmoid_rampup(epoch, 100)) * np.log(2)
            mask = (uncertainty < threshold).float()
            the_three_features__consistency_loss = torch.sum(
                mask * the_three_consistency_dist) / (2 * torch.sum(mask) + 1e-16)

            T = 4
            _, c, w, h = the_four_features__preds_.shape
            volume_batch_r = the_four_features__preds_.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            the_four_features__preds = torch.zeros([stride * T, c,w, h]).cuda()
            for i in range(T // 2):
                the_four_features__preds[2*stride * i: 2*stride * (i + 1)] = volume_batch_r
            the_four_features__preds = F.softmax(the_four_features__preds, dim=1)
            the_four_features__preds = the_four_features__preds.reshape(T, stride, c,w, h)
            the_four_features__preds = torch.mean(the_four_features__preds, dim=0)
            uncertainty = -1.0 * \
                          torch.sum(the_four_features__preds * torch.log(the_four_features__preds + 1e-6), dim=1,
                                    keepdim=True)
            the_four_consistency_dist = F.huber_loss(
                the_four_features, the_four_features_, delta=1.0)
            threshold = (0.5 + 0.5 * sigmoid_rampup(epoch, 100)) * np.log(2)
            mask = (uncertainty < threshold).float()
            the_four_features__consistency_loss = torch.sum(
                mask * the_four_consistency_dist) / (2 * torch.sum(mask) + 1e-16)
            #consistency_standard和consistency_loss_out可以选择使用，一个是标准L_C，一个是计算了不确定性
            consistency_loss = consistency_weight *(consistency_standard + low_level_features_consistency_loss+the_two_features_consistency_loss+the_three_features__consistency_loss+the_four_features__consistency_loss)
            # consistency_loss = consistency_weight * consistency_loss_out
            #########################################CTSA###########################################
            #########这里可以只对最后一层进行CTSA，也可以对全部层进行CTSA###########################
            attnlist=[]
            for s,t in zip(s_encoder,t_encoder):
                s_encoder_label, s_encoder_unlabel = s.split([num_lb, num_ulb])
                _,embedding_channels, _, _ = s_encoder_unlabel.shape
                atattn = CTSA(num_heads=3,
                                embedding_channels=embedding_channels,
                                channel_num=embedding_channels,
                                channel_num_out=embedding_channels,
                                attention_dropout_rate=0.1,
                                patch_num=1444,
                                num_class=num_classes)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                atattn.to(device)
                st_input=torch.cat((s_encoder_unlabel, t))
                self_attn_u=atattn(st_input)
                attnlist.append(self_attn_u)
            c1, c2, c3, c4=attnlist
            attn_output = model_train.module.decode_head(c1, c2, c3, c4)
            #########如果只在c4加CTSA###########
            # c1_, c2_, c3_, c4_ = s_encoder
            # attn_output = model_train.module.decode_head(c1_.split([num_lb, num_ulb])[0],
            #                                              c2_.split([num_lb, num_ulb])[0],
            #                                              c3_.split([num_lb, num_ulb])[0], c4)
            ###################################
            _,_,W,H=imgs_unlabel.shape
            attn_output = F.interpolate(attn_output, size=(W, H), mode='bilinear', align_corners=True)
            attn_loss = criterion_u(attn_output, pseudo_label)*0.25
            #######################################################################################
            #----------------------#
            #   计算损失
            #----------------------#
            loss = suloss  + consistency_loss + attn_loss
            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()
            update_ema_variables(model_train, ema_model, 0.99, epoch)


        else:
            print('else')
        total_loss      += loss.item()
        suloss_item  += suloss.item()
        consistency_loss_item += consistency_loss.item()
        attn_loss_item += attn_loss.item()
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # ----------------------#
            #   前向传播
            # ----------------------#
            _,_,_,_,outputs,s_encoder = model_train(imgs)
            # ----------------------#
            #   计算损失
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    # 'val_loss_at': val_loss_at / (iteration + 1),
                                    'f_score': val_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    print('total Loss: %.6f' % (total_loss / epoch_step))
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        # eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('su Loss: %.6f ||u Loss: %.6f ||consistency Loss: %.6f || attn Loss: %.6f || Total Loss: %.6f ||Val Loss: %.3f ' % (suloss_item / epoch_step, loss_u / epoch_step, consistency_loss_item / epoch_step, attn_loss_item / epoch_step,total_loss / epoch_step, val_loss / epoch_step_val))


        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        # if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        #     torch.save(ema_model.state_dict(), os.path.join(save_dir, 'ema_ep%03d-loss%.3f-val_loss_at%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss_at / epoch_step_val)))


        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
