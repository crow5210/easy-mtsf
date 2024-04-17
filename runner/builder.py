
import time
import torch
from tqdm import tqdm
import numpy as np
from utils.metrics import masked_mae
from utils.meter_pool import MeterPool
from utils.checkpoint import save_best_model,save_model
from data.transform import re_standard_transform
from utils.plot import plot_multi_result


from typing import Dict

from torch import nn, optim
from torch.optim import lr_scheduler


def build_optim(optim_cfg: Dict, model: nn.Module) -> optim.Optimizer:
    """Build optimizer from `optim_cfg`
    `optim_cfg` is part of config which defines fields about optimizer

    structure of `optim_cfg` is
    {
        'TYPE': (str or type) optimizer name or type, such as ``Adam``, ``SGD``,
            or custom optimizer type.
        'PARAM': (Dict) optimizer init params except first param `params`
    }

    Note:
        Optimizer is initialized by reflection, please ensure optim_cfg['TYPE'] is in `torch.optim`

    Examples:
        optim_cfg = {
            'TYPE': 'Adam',
            'PARAM': {
                'lr': 1e-3,
                'betas': (0.9, 0.99)
                'eps': 1e-8,
                'weight_decay': 0
            }
        }
        An `Adam` optimizer will be built.

    Args:
        optim_cfg (Dict): optimizer config
        model (nn.Module): model defined by user

    Returns:
        optimizer (optim.Optimizer)
    """

    optim_type = getattr(optim, optim_cfg['TYPE'])
    optim_param = optim_cfg['PARAM'].copy()
    optimizer = optim_type(model.parameters(), **optim_param)
    return optimizer


def build_lr_scheduler(lr_scheduler_cfg: Dict, optimizer: optim.Optimizer) -> lr_scheduler._LRScheduler:
    """Build lr_scheduler from `lr_scheduler_cfg`
    `lr_scheduler_cfg` is part of config which defines fields about lr_scheduler

    structure of `lr_scheduler_cfg` is
    {
        'TYPE': (str or type) lr_scheduler name or type, such as ``MultiStepLR``, ``CosineAnnealingLR``,
            or custom lr_scheduler type
        'PARAM': (Dict) lr_scheduler init params except first param `optimizer`
    }

    Note:
        LRScheduler is initialized by reflection, please ensure
        lr_scheduler_cfg['TYPE'] is in `torch.optim.lr_scheduler` or `easytorch.easyoptim.easy_lr_scheduler`,
        if the `type` is not found in `torch.optim.lr_scheduler`,
        it will continue to be search in `easytorch.easyoptim.easy_lr_scheduler`

    Examples:
        lr_scheduler_cfg = {
            'TYPE': 'MultiStepLR',
            'PARAM': {
                'milestones': [100, 200, 300],
                'gamma': 0.1
            }
        }
        An `MultiStepLR` lr_scheduler will be built.

    Args:
        lr_scheduler_cfg (Dict): lr_scheduler config
        optimizer (nn.Module): optimizer

    Returns:
        LRScheduler
    """

    lr_scheduler_cfg['TYPE'] = lr_scheduler_cfg['TYPE']
    scheduler_type = getattr(lr_scheduler, lr_scheduler_cfg['TYPE'])

    scheduler_param = lr_scheduler_cfg['PARAM'].copy()
    scheduler_param['optimizer'] = optimizer
    scheduler = scheduler_type(**scheduler_param)
    return scheduler



def build_model(cfg_general,cfg_model): 
    models = {}
    models["name"] = cfg_model['NAME']
    models["model"] = cfg_model["ARCH"](**cfg_model["PARAM"])
    models["optimizer"] = build_optim(cfg_general['OPTIM'], models["model"])
    models["scheduler"] = build_lr_scheduler(cfg_general['LR_SCHEDULER'], models["optimizer"])
    models["clip_grad"] = cfg_general.get('CLIP_GRAD_PARAM',None)
    models["cl_param"] = cfg_general.get('CL_PARAM',None)
    models["forward_features"] = cfg_model.get("FORWARD_FEATURES", [0,1,2])
    models["target_features"] = cfg_model.get("TARGET_FEATURES", [0])
    models["loss"] = cfg_model.get("LOSS",masked_mae)
    return models



def build_meter(meters,metrics,meter_type,horizon=None):
    meters.register(f"{meter_type}_time",'{:.2f} (s)', plt=False)
    for key in metrics.keys():
        if meter_type == "test" and isinstance(horizon,list):
            for h in horizon:
                meters.register(f"{meter_type}_{key}_{h}","{:.3f}", plt=False)
            meters.register(f"{meter_type}_{key}_avg","{:.3f}", plt=True if key == "MAE" and  meter_type =="train" else False)
        else:
            meters.register(f"{meter_type}_{key}","{:.3f}", plt=True if key == "MAE" and  meter_type =="train" else False)

 
    if meter_type=="train":
        meters.register("lr",'{:.2e}', plt=False)


def curriculum_learning(args, epoch = None):
    if epoch is None:
        return args.prediction_length
    epoch -= 1
    # generate curriculum length
    if epoch < args.warm_up_epochs:
        # still warm up
        cl_length = args.prediction_length
    else:
        _ = ((epoch - args.warm_up_epochs) // args.cl_epochs + 1) * args.cl_step_size
        cl_length = min(_, args.prediction_length)
    return cl_length



def epoch_iter(models,datasets,epoch,mode,metrics,horizons,logger,tensbd,device, out_dir=None):
    assert mode in ['train','val','test'],"mode must be chosen in train,val or test"

    forward_features = models["forward_features"]
    target_features = models["target_features"]
    data_loader = datasets[mode]
    scaler = datasets["scaler"]

    model = models["model"]
    meters = MeterPool()
    build_meter(meters,metrics,mode,horizons)

    if mode=="train":
        scheduler = models["scheduler"]
        meters.update('lr', scheduler.get_last_lr()[0])

    if mode=="test":
        reals = []
        preds = []

    test_start_time = time.time()
    for (future_data, history_data) in tqdm(data_loader):
        history_data = history_data.to(device)
        future_data = future_data.to(device)
            
        history_data = history_data[:,:,:,forward_features]
        future_data_4_dec = future_data[:,:,:,forward_features]
        prediction_data = model(history_data=history_data, future_data=future_data_4_dec, batch_seen=1, epoch=None, train=False)
        prediction = re_standard_transform(prediction_data[:,:,:,target_features],**scaler["args"])
        real_value = re_standard_transform(future_data[:,:,:,target_features],**scaler["args"])
        # prediction = prediction_data[:,:,:,target_features]
        # real_value = future_data[:,:,:,target_features]
        if mode=="train":
            optimizer = models["optimizer"]
            clip_grad_param = models["clip_grad"]
            cl_param = models["cl_param"]
            if cl_param:
                cl_length = curriculum_learning(cl_param,epoch=epoch)
                prediction = prediction[:, :cl_length, :, :]
                real_value = real_value[:, :cl_length, :, :]

            loss = models["loss"](prediction,real_value)
            optimizer.zero_grad()
            loss.backward()
            if clip_grad_param:
                torch.nn.utils.clip_grad_norm_(model.parameters(), **clip_grad_param)
            optimizer.step()

        if mode=="test":
            reals.append(real_value[...,0].detach().cpu())
            preds.append(prediction[...,0].detach().cpu())

        for metric_name, metric_func in metrics.items():
            if mode=="test" and isinstance(horizons,list):
                for h in horizons:
                    metric_item = metric_func(*[prediction[:,h-1,:,:], real_value[:,h-1,:,:]])
                    meters.update(f"{mode}_{metric_name}_{h}", metric_item.item())
                metric_item = metric_func(*[prediction, real_value])
                meters.update(f"{mode}_{metric_name}_avg", metric_item.item())
            else:
                metric_item = metric_func(*[prediction, real_value])
                meters.update(f"{mode}_{metric_name}", metric_item.item())

    test_end_time = time.time()
    meters.update(f"{mode}_time", test_end_time - test_start_time)
    meters.print_meters(logger)
    meters.plt_meters(epoch,tensbd)

    if mode=="train":
        scheduler.step()

    if mode=="test":
        reals = torch.cat(reals,axis=0)
        preds = torch.cat(preds,axis=0)
        if isinstance(horizons,list):
            plot_multi_result(reals,preds,f"MAE of {models['name']}:{meters.get_avg(f'{mode}_MAE_avg'):.3f}",out_dir = out_dir)
        else:
            plot_multi_result(reals,preds,f"MAE of {models['name']}:{meters.get_avg(f'{mode}_MAE'):.3f}",out_dir = out_dir)

    return meters
    


def inference(models, datasets, metrics, horizons, device, logger=None, out_dir=None):
    logger.info('start test')
    models["model"].eval()
    with torch.no_grad():
        meters = epoch_iter(models,datasets,0,"test",metrics, horizons ,logger,None, device, out_dir)
    return meters

def validate(models, datasets, epoch, best_metrics, ckpt_save_dir, metrics, horizons , device,logger=None, tensbd=None):
    models["model"].eval()
    with torch.no_grad():
        meters = epoch_iter(models,datasets,epoch,"val",metrics, horizons ,logger,tensbd,device)
    save_best_model(models,meters,best_metrics,ckpt_save_dir,epoch,logger=logger)
    return meters

def fit(models, datasets, epoch, best_metrics, ckpt_save_dir, metrics, horizons, device, logger=None, tensbd=None):   
    models["model"].train()
    meters = epoch_iter(models,datasets,epoch,"train", metrics, horizons ,logger,tensbd,device)
    save_model(models,best_metrics,epoch,ckpt_save_dir)
    return meters


