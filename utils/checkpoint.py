import os,re
import torch
import glob

# ------------------checkpoints ----------------------------------------------------------------
def backup_last_ckpt(last_ckpt_path, epoch):
    if epoch > 1:
        os.rename(last_ckpt_path, last_ckpt_path + '.bak')

def clear_ckpt(ckpt_save_dir, name_pattern = '*.pt.bak'):
    ckpt_list = glob.glob(os.path.join(ckpt_save_dir, name_pattern))
    for ckpt in ckpt_list:
        os.remove(ckpt)

def get_ckpt_path(epoch,ckpt_save_dir,model_name):
    epoch_str = str(epoch).zfill(3)
    ckpt_name = '{}_{}.pt'.format(model_name, epoch_str)
    return os.path.join(ckpt_save_dir, ckpt_name)

def save_model(models,best_metrics,epoch,ckpt_save_dir):
    ckpt_dict = {
        'epoch': epoch,
        'model_state_dict': models["model"].state_dict(),
        'optim_state_dict': models["optimizer"].state_dict(),
        'best_metrics': best_metrics
    }

    # backup last epoch
    last_ckpt_path = get_ckpt_path(epoch - 1,ckpt_save_dir,models["name"])
    backup_last_ckpt(last_ckpt_path, epoch)

    # save ckpt
    ckpt_path = get_ckpt_path(epoch,ckpt_save_dir,models["name"])
    torch.save(ckpt_dict, ckpt_path)

    clear_ckpt(ckpt_save_dir)

def save_best_model(models,meters,best_metrics,ckpt_save_dir,epoch,greater_best=True,metric_name="val_MAE",logger=None):
    metric = meters.get_avg(metric_name)
    best_metric = best_metrics.get(metric_name)

    if best_metric is None or (metric < best_metric if greater_best else metric > best_metric):
        logger.info(f"save best model:epoch {epoch} val_MAE {metric:.3f}")
        best_metrics[metric_name] = metric
        model = models["model"]
        optimizer = models["optimizer"]

        ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'best_metrics': best_metrics
        }
        
        ckpt_path = os.path.join(
            ckpt_save_dir,
            '{}_best_{}.pt'.format(models["name"], metric_name.replace('/', '_'))
        )
        torch.save(ckpt_dict, ckpt_path)

def get_last_ckpt_path(ckpt_save_dir, name_pattern = r'^.+_[\d]*.pt$'):
    ckpt_list = [f for f in os.listdir(ckpt_save_dir) if re.search(name_pattern, f) is not None]
    ckpt_list.sort()
    return os.path.join(ckpt_save_dir, ckpt_list[-1])

def resume_model(ckpt_save_dir,ckpt_path=None):    
    checkpoint_dict = None
    try:
        if ckpt_path is None:
            ckpt_path = get_last_ckpt_path(ckpt_save_dir)
        else:
            ckpt_path = os.path.join(ckpt_save_dir,ckpt_path)
        checkpoint_dict = torch.load(ckpt_path)
    except (IndexError, OSError, KeyError):
        pass
    return checkpoint_dict