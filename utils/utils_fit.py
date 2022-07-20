import os

import torch
from tqdm import tqdm

from .utils import get_lr, show_result
from .utils_metrics import PSNR, SSIM


def fit_one_epoch(G_model_train, D_model_train, G_model, D_model, VGG_feature_model, loss_history, G_optimizer, D_optimizer, BCE_loss, MSE_loss, 
                epoch, epoch_step, gen, Epoch, cuda, fp16, scaler, save_period, save_dir, photo_save_step, local_rank=0):
    G_total_loss = 0
    D_total_loss = 0
    G_total_PSNR = 0
    G_total_SSIM = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        
        lr_images, hr_images = batch
        batch_size      = lr_images.size()[0]
        y_real, y_fake  = torch.ones(batch_size), torch.zeros(batch_size)
        
        with torch.no_grad():
            if cuda:
                lr_images, hr_images, y_real, y_fake  = lr_images.cuda(local_rank), hr_images.cuda(local_rank), y_real.cuda(local_rank), y_fake.cuda(local_rank)
        
        if not fp16:
            #-------------------------------------------------#
            #   训练判别器
            #-------------------------------------------------#
            D_optimizer.zero_grad()

            D_result                = D_model_train(hr_images)
            D_real_loss             = BCE_loss(D_result, y_real)
            D_real_loss.backward()

            G_result                = G_model_train(lr_images)
            D_result                = D_model_train(G_result).squeeze()
            D_fake_loss             = BCE_loss(D_result, y_fake)
            D_fake_loss.backward()

            D_optimizer.step()

            D_train_loss            = D_real_loss + D_fake_loss

            #-------------------------------------------------#
            #   训练生成器
            #-------------------------------------------------#
            G_optimizer.zero_grad()

            G_result                = G_model_train(lr_images)
            image_loss              = MSE_loss(G_result, hr_images)

            D_result                = D_model_train(G_result).squeeze()
            adversarial_loss        = BCE_loss(D_result, y_real)

            perception_loss         = MSE_loss(VGG_feature_model(G_result), VGG_feature_model(hr_images))

            G_train_loss            = image_loss + 1e-3 * adversarial_loss + 2e-6 * perception_loss 

            G_train_loss.backward()
            G_optimizer.step()
        else:
            from torch.cuda.amp import autocast
            
            #-------------------------------------------------#
            #   训练判别器
            #-------------------------------------------------#
            with autocast():
                D_optimizer.zero_grad()
                D_result                = D_model_train(hr_images)
                D_real_loss             = BCE_loss(D_result, y_real)
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(D_real_loss).backward()
            
            with autocast():
                G_result                = G_model_train(lr_images)
                D_result                = D_model_train(G_result).squeeze()
                D_fake_loss             = BCE_loss(D_result, y_fake)
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(D_fake_loss).backward()
            scaler.step(D_optimizer)
            scaler.update()
            
            D_train_loss            = D_real_loss + D_fake_loss
            #-------------------------------------------------#
            #   训练生成器
            #-------------------------------------------------#
            with autocast():
                G_optimizer.zero_grad()
                G_result                = G_model_train(lr_images)
                image_loss              = MSE_loss(G_result, hr_images)

                D_result                = D_model_train(G_result).squeeze()
                adversarial_loss        = BCE_loss(D_result, y_real)

                perception_loss         = MSE_loss(VGG_feature_model(G_result), VGG_feature_model(hr_images))

                G_train_loss            = image_loss + 1e-3 * adversarial_loss + 2e-6 * perception_loss 
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(G_train_loss).backward()
            scaler.step(G_optimizer)
            scaler.update()
            
        G_total_loss            += G_train_loss.item()
        D_total_loss            += D_train_loss.item()

        with torch.no_grad():
            G_total_PSNR        += PSNR(G_result, hr_images).item()
            G_total_SSIM        += SSIM(G_result, hr_images).item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'G_loss'    : G_total_loss / (iteration + 1), 
                                'D_loss'    : D_total_loss / (iteration + 1), 
                                'G_PSNR'    : G_total_PSNR / (iteration + 1), 
                                'G_SSIM'    : G_total_SSIM / (iteration + 1), 
                                'lr'        : get_lr(G_optimizer)})
            pbar.update(1)

            if iteration % photo_save_step == 0:
                show_result(epoch + 1, G_model, lr_images, hr_images)

    G_total_loss = G_total_loss / epoch_step
    D_total_loss = D_total_loss / epoch_step
    
    if local_rank == 0:
        pbar.close()
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('G Loss: %.4f || D Loss: %.4f ' % (G_total_loss, D_total_loss))
        loss_history.append_loss(epoch + 1, G_total_loss = G_total_loss, D_total_loss = D_total_loss, G_total_PSNR = G_total_PSNR, G_total_SSIM = G_total_SSIM)

        #----------------------------#
        #   每若干个世代保存一次
        #----------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(G_model.state_dict(), os.path.join(save_dir, 'G_Epoch%d-GLoss%.4f-DLoss%.4f.pth'%(epoch + 1, G_total_loss, D_total_loss)))
            torch.save(D_model.state_dict(), os.path.join(save_dir, 'D_Epoch%d-GLoss%.4f-DLoss%.4f.pth'%(epoch + 1, G_total_loss, D_total_loss)))
            
        torch.save(G_model.state_dict(), os.path.join(save_dir, "G_model_last_epoch_weights.pth"))
        torch.save(D_model.state_dict(), os.path.join(save_dir, "D_model_last_epoch_weights.pth"))
