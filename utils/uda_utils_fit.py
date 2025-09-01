import os

import torch
from tqdm import tqdm

from utils.utils import get_lr
import torch.nn as nn



def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, 
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0, 
                  file_name='DAUB_to_DAUB', input_shape=[544, 544], val_each_epoch=10):
    loss = 0
    val_loss = 0
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()

    for iteration, (souce_batch, target_batch) in enumerate(gen):
        if iteration >= epoch_step:
            break
        
        source_images, source_target, source_box = souce_batch[0], souce_batch[1], souce_batch[2]
        
        target_images, target_target = target_batch[0], target_batch[1]

        b, c, t, h, w = source_images.shape

        with torch.no_grad():
            if cuda:
                source_images = source_images.cuda(local_rank)
                target_images = target_images.cuda(local_rank)

                target_target = [ann.cuda(local_rank) for ann in target_target]

        optimizer.zero_grad()
        if not fp16:  # not fp16 = True
            target_outputs = model_train(source_images[:, :, -1, :, :], source_box, target_images)

            # ###############
            yololoss = yolo_loss(target_outputs, target_target)

            # ############### 
            loss_value = yololoss

            loss_value.backward()
            optimizer.step()
            
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                loss_value = yolo_loss(outputs, targets)
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)
        loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'loss' : loss / (iteration + 1), 
                                'yololoss' : yololoss.item(), 
                                'lr'  : get_lr(optimizer)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()


    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        if (epoch  + 1) % val_each_epoch == 0:
            eval_callback.on_epoch_end(epoch + 1, model_train_eval)

        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))

        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Train Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))