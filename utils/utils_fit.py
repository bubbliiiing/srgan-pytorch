import torch
from tqdm import tqdm

from .utils import show_result, get_lr
from .utils_metrics import PSNR, SSIM


def fit_one_epoch(G_model_train, D_model_train, G_model, D_model, VGG_feature_model, G_optimizer, D_optimizer, BCE_loss, MSE_loss, epoch, epoch_size, gen, Epoch, cuda, batch_size, save_interval):
    G_total_loss = 0
    D_total_loss = 0
    G_total_PSNR = 0
    G_total_SSIM = 0

    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            
            with torch.no_grad():
                lr_images, hr_images    = batch
                lr_images, hr_images    = torch.from_numpy(lr_images).type(torch.FloatTensor), torch.from_numpy(hr_images).type(torch.FloatTensor)
                y_real, y_fake          = torch.ones(batch_size), torch.zeros(batch_size)
                if cuda:
                    lr_images, hr_images, y_real, y_fake  = lr_images.cuda(), hr_images.cuda(), y_real.cuda(), y_fake.cuda()
            
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
            
            G_total_loss            += G_train_loss.item()
            D_total_loss            += D_train_loss.item()

            with torch.no_grad():
                G_total_PSNR        += PSNR(G_result, hr_images).item()
                G_total_SSIM        += SSIM(G_result, hr_images).item()
                
            pbar.set_postfix(**{'G_loss'    : G_total_loss / (iteration + 1), 
                                'D_loss'    : D_total_loss / (iteration + 1), 
                                'G_PSNR'    : G_total_PSNR / (iteration + 1), 
                                'G_SSIM'    : G_total_SSIM / (iteration + 1), 
                                'lr'        : get_lr(G_optimizer)})
            pbar.update(1)

            if iteration % save_interval == 0:
                show_result(epoch + 1, G_model_train, lr_images, hr_images)

    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('G Loss: %.4f || D Loss: %.4f ' % (G_total_loss / epoch_size, D_total_loss / epoch_size))
    print('Saving state, iter:', str(epoch+1))
    
    if (epoch + 1) % 10==0:
        torch.save(G_model.state_dict(), 'logs/G_Epoch%d-GLoss%.4f-DLoss%.4f.pth'%((epoch + 1), G_total_loss / epoch_size, D_total_loss / epoch_size))
        torch.save(D_model.state_dict(), 'logs/D_Epoch%d-GLoss%.4f-DLoss%.4f.pth'%((epoch + 1), G_total_loss / epoch_size, D_total_loss / epoch_size))
