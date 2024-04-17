from plotly.subplots import make_subplots
import torch
import plotly.graph_objects as go
import plotly.offline as offline

def plot_multi_result(reals,preds,title,plot_series_num=4,out_dir=None):
    reals = torch.cat([reals[0,:-1,:],reals[:,-1,:]])

    if preds.shape[1]==1:
        plot_series_length=100
    else:
        plot_series_length = 10
    plot_series_num = min(plot_series_num,preds.shape[-1])
    total_length = plot_series_length*preds.shape[1]

    # total_length = reals.shape[0]

    fig = make_subplots(rows=plot_series_num,cols=1)
    for i in range(plot_series_num):
        fig.append_trace(go.Scatter(y=reals[:min(reals.shape[0],total_length),i],name="real"),i+1,1)
        for j in range(0,min(preds.shape[0],total_length),preds.shape[1]):
            fig.append_trace(go.Scatter(x=list(range(j,j+preds.shape[1])),y=preds[j,:,i]),i+1,1)

    if out_dir is not None :
        fn = title.split(":")[0].split(" ")[-1]
        fig.update_layout(title_text=title)
        offline.plot(fig, filename=f'{out_dir}/{fn}.html', auto_open=False)
    else:
        fig.update_layout(height=max(100*plot_series_num,400), width=800, title_text=title)
        fig.show()


def plot_multi_result_darts(reals,preds,title,plot_series_num=4,save_dir=None):
    plot_series_num = min(3,preds.shape[-1])

    fig = make_subplots(rows=plot_series_num,cols=1)
    for i in range(plot_series_num):
        fig.append_trace(go.Scatter(y=reals[i]._xa.values[:,0,0][:preds.shape[0]*preds.shape[1]],name="real"),i+1,1)
        for j in range(preds.shape[0]):
            fig.append_trace(go.Scatter(x=list(range(preds.shape[1]*j,preds.shape[1]*(j+1))),y=preds[j,:,i],name="pred"),i+1,1)
            
    if save_dir is None:
        fig.update_layout(height=max(100*plot_series_num,400), width=800, title_text=title)
        fig.show()
    else:
        offline.plot(fig, filename=f'{save_dir}/{title}.html', auto_open=False)   