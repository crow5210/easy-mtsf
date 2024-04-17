import os,time,sys
sys.path.append(os.path.abspath(__file__ + "/../.."))
from utils.__init__ import config_md5,config_str
from utils.logger import init_tensorboard,init_logger
from utils.checkpoint import resume_model
from runner.builder import build_model,validate,fit,inference
from data.data_loader import build_dataset
from config.cfg_algorithm import CFG_MODEL
from config.cfg_general import CFG_GENERAL
from config.cfg_dataset import CFG_DATASET
from utils.__init__ import count_parameters
from utils.gen_report import result_decode,gen_latex_table_code,gen_report_csv,gen_ranking_graph,gen_polar_graph
import torch
import plotly.express as px
from utils.metrics import masked_mae
import pandas as pd
import plotly.offline as offline

def visualize_result(algos,epochs,out_dir,device,data = "Electricity"):
    hist = []
    futu = []
    pred = []
    index = []
    method = []
    
    for algo in algos:
        model_args = CFG_MODEL[algo](data, CFG_DATASET[data]["NUM_NODES"],
                                CFG_GENERAL.DATASET.HISTORY_SEQ_LEN,
                                CFG_GENERAL.DATASET.FUTURE_SEQ_LEN,
                                not CFG_GENERAL.DATASET.NORM_EACH_CHANNEL,
                                CFG_DATASET[data].STEPS_PER_DAY)
        
        md5 = config_md5(CFG_GENERAL, CFG_DATASET[data], model_args)
        ckpt_save_dir = os.path.join(out_dir, "_".join([data, algo, str(epochs)]),md5)

        models = build_model(CFG_GENERAL, model_args)
        datasets = build_dataset(CFG_GENERAL, CFG_DATASET[data])
        models["model"].to(device)

        ckpt_path = f'{algo}_{data}_best_val_MAE.pt'
        checkpoint_dict = resume_model(ckpt_save_dir,ckpt_path)
        if checkpoint_dict is not None:            
            models["model"].load_state_dict(checkpoint_dict['model_state_dict'], strict=True)
            models["model"].eval()
            with torch.no_grad():
                mode = "test"
                forward_features = models["forward_features"]
                data_loader = datasets[mode]
                scaler = datasets["scaler"]
                model = models["model"]

                future_data, history_data = next(iter(data_loader))
                len_history = history_data.shape[1]
                len_future = future_data.shape[1]
                history_data = history_data.to(device)
                future_data = future_data.to(device)
                    
                history_data = history_data[:,:,:,forward_features]
                future_data_4_dec = future_data[:,:,:,forward_features]

                prediction_data = model(history_data=history_data, future_data=future_data_4_dec, batch_seen=1, epoch=None, train=False)
                mae = masked_mae(prediction_data[...,0],future_data[...,0])


                hist.append(torch.cat([history_data[0,:,3,0].detach().cpu(),torch.tensor([torch.nan]*len_future)]))
                pred.append(torch.cat([torch.tensor([torch.nan]*len_history),prediction_data[0,:,3,0].detach().cpu()]))
                futu.append(torch.cat([torch.tensor([torch.nan]*len_history),future_data[0,:,3,0].detach().cpu()]))
                method += [f"mae of {algo}:{mae:.3f}"]*(len_history+len_future)
                index.append(torch.arange(0,len_history+len_future,1))
        else:
            print(f"can't find ckpt:{ckpt_path}")

    df = pd.DataFrame({"index":torch.cat(index),
                    "method":method,
                        "future":torch.cat(futu),
                        "prediction":torch.cat(pred),
                        "history":torch.cat(hist)})


    fig = px.line(df,
                x="index",
                y=["history","prediction","future"],
                line_group='method',
                facet_col='method',
                facet_col_wrap = 2,
    # #               log_y=True,
                )
    fig.update_traces(mode='lines')
    # fig.show()
    report_dir = f"{out_dir}/report"
    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)
    offline.plot(fig, filename=f'{report_dir}/visualization.html', auto_open=False)


def train_test(datas,algos,epochs,val_interval,out_dir,device):
    result_total = {}
    for algo in algos:
        for data in datas:
            if isinstance(data, list):
                source = data[0]
                target = data[1]
            else:
                source = data
                target = data

            model_args = CFG_MODEL[algo](source, CFG_DATASET[source]["NUM_NODES"],
                                  CFG_GENERAL.DATASET.HISTORY_SEQ_LEN,
                                  CFG_GENERAL.DATASET.FUTURE_SEQ_LEN,
                                  not CFG_GENERAL.DATASET.NORM_EACH_CHANNEL,
                                  CFG_DATASET[source].STEPS_PER_DAY)
            
            md5 = config_md5(CFG_GENERAL, CFG_DATASET[source], model_args)
            ckpt_save_dir = os.path.join(out_dir, "_".join([source, algo, str(epochs)]),md5)
            if not os.path.isdir(ckpt_save_dir):
                os.makedirs(ckpt_save_dir)
            logger = init_logger(f"{source}_{algo}_{epochs}",ckpt_save_dir)
            tensbd = init_tensorboard(ckpt_save_dir)
            logger.info(f"run {source}_{algo}")
            
            models = build_model(CFG_GENERAL, model_args)
            datasets = build_dataset(CFG_GENERAL, CFG_DATASET[source])

            logger.info(f"parameter count of {algo}:{count_parameters(models['model'])}")
            logger.info(f"model config: \n{config_str(model_args)}")
            logger.info(f"dataset config:  \n{config_str(CFG_DATASET[source])}")
            models["model"].to(device)
            
            best_metrics = {}
            start_epoch = 0
            checkpoint_dict = resume_model(ckpt_save_dir)
            if checkpoint_dict is not None:
                logger.info('Loading Checkpoint from \'{}\''.format(ckpt_save_dir))
                models["model"].load_state_dict(checkpoint_dict['model_state_dict'], strict=True)
                models["optimizer"].load_state_dict(checkpoint_dict['optim_state_dict'])
                models["scheduler"].last_epoch = checkpoint_dict['epoch']
                best_metrics = checkpoint_dict['best_metrics']
                start_epoch = checkpoint_dict['epoch']

            logger.info('Initializing training.')
            for epoch in range(start_epoch,epochs):
                epoch = epoch + 1
                logger.info('Epoch {:d} / {:d}'.format(epoch, epochs))
                fit(models, datasets, epoch, best_metrics, ckpt_save_dir, CFG_GENERAL.METRICS,None, device, logger, tensbd)
                if epoch % val_interval == 0:
                    validate(models, datasets, epoch, best_metrics, ckpt_save_dir, CFG_GENERAL.METRICS,None, device, logger, tensbd)
            logger.info('The training finished at {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
            
            # test
            if source != target:
                datasets = build_dataset(CFG_GENERAL, CFG_DATASET[target])
            ckpt_path = f'{algo}_{source}_best_val_MAE.pt'
            checkpoint_dict = resume_model(ckpt_save_dir,ckpt_path)
            if checkpoint_dict is not None:            
                models["model"].load_state_dict(checkpoint_dict['model_state_dict'], strict=True)
                result_total[f"{source}|{target}_{algo}"] = inference(models, datasets, CFG_GENERAL.METRICS, CFG_GENERAL.TEST.get("HORIZON",None), device, logger, ckpt_save_dir)
            else:
                logger.info(f"can't find ckpt:{ckpt_path}")

            tensbd.close()

    report = result_decode(result_total,CFG_GENERAL.METRICS.keys(),CFG_GENERAL.TEST.get("HORIZON",None))
    gen_latex_table_code(report,out_dir,CFG_GENERAL.TEST.get("HORIZON",None))
    report_df = gen_report_csv(report,out_dir)
    gen_ranking_graph(report_df,out_dir)
    gen_polar_graph(report_df,out_dir)
    return result_total


