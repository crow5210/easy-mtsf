import pandas as pd
import plotly.express as px
import plotly.offline as offline
import os

def result_decode(meters,metrics,horizon):
    report = {}
    for k in meters.keys():
        data,_ = k.split("_")
        report[data] = {}
        report[data]["ALGOS"] = []
        report[data]["SPEED"] = []
        if isinstance(horizon,list):
            report[data].update({metric:[[] for _ in horizon+["avg"]] for metric in metrics})
        else:
            report[data].update({metric:[] for metric in metrics})
        


    for k,v in meters.items():
        data,algo = k.split("_")
        report[data]["ALGOS"].append(algo)
        report[data]["SPEED"].append(f'{v.get_avg("test_time"):.2f}')
        for metric in metrics:
            if isinstance(horizon,list):
                for i,h in enumerate(horizon+["avg"]):
                    report[data][metric][i].append(f'{v.get_avg(f"test_{metric}_{h}"):.3f}')
            else:
                report[data][metric].append(f'{v.get_avg(f"test_{metric}"):.3f}')

    return report

def latex_table_head_multi_col(report,horizon):
    algos = report[list(report.keys())[0]]["ALGOS"]

    str1 = f"""
% Please add the following required packages to your document preamble:
% \\usepackage{{multirow}}
\\begin{{table}}[]
\\begin{{tabular}}{{l|l|{"|".join(["cccc" for _ in algos])}|}}
\\hline
"""

    str2 = "\multirow{2}{*}{data} & \multirow{2}{*}{metric} "
    for algo in algos:
        str2 += f"& \multicolumn{{4}}{{c|}}{{{algo}}} "
    str2+= "\\\\\n"  
    
    str2 += "                      &        "
    for algo in algos:
        for h in horizon+['avg']:
            str2 += f"& {h}       "
    str2 += "\\\\ \\hline"

    return str1+str2+"\n"


def latex_table_head_single_col(report):
    algos = report[list(report.keys())[0]]["ALGOS"]

    str1 = f"""
% Please add the following required packages to your document preamble:
% \\usepackage{{multirow}}
\\begin{{table}}[]
\\begin{{tabular}}{{l|l|{"|".join(["c" for _ in algos])}|}}
\\hline
"""

    str2 = " data & metric "
    for algo in algos:
        str2 += f"& {algo} "
    str2 += "\\\\ \\hline"

    return str1+str2+"\n"


def latex_table_tail():
    return """
\\end{tabular}
\\end{table}
"""

def latex_table_body_multi_col(data,ret,horizon):
    horizon = horizon+["avg"]
    str_body = f"\\multirow{{4}}{{*}}{{{data}}}  & MAE    "
    for i in range(len(horizon)):
        for j in range(len(ret["ALGOS"])):
            str_body += f"& {ret['MAE'][i][j]}"
    str_body += "\\\\\n"

    str_body += f"                      & MAPE   "
    for i in range(len(horizon)):
        for j in range(len(ret["ALGOS"])):
            str_body += f"& {ret['MAPE'][i][j]}"
    str_body += "\\\\\n"

    str_body += f"                      & WAPE   "
    for i in range(len(horizon)):
        for j in range(len(ret["ALGOS"])):
            str_body += f"& {ret['WAPE'][i][j]}"
    str_body += f"\\\\ \\cline{{2-{2+len(ret['ALGOS'])*len(horizon)}}}\n"

    str_body += f"                      & SPEED   "
    for i in range(len(ret["ALGOS"])):
        str_body += f"&\multicolumn{{4}}{{c|}} {{{ret['SPEED'][i]}}}"
    str_body += "\\\\ \\hline\n"

    return str_body

def latex_table_body_single_col(data,ret):
    str_body = f"\\multirow{{4}}{{*}}{{{data}}}  & MAE    "
    for j in range(len(ret["ALGOS"])):
        str_body += f"& {ret['MAE'][j]}"
    str_body += "\\\\\n"

    str_body += f"                      & MAPE   "
    for j in range(len(ret["ALGOS"])):
        str_body += f"& {ret['MAPE'][j]}"
    str_body += "\\\\\n"

    str_body += f"                      & WAPE   "
    for j in range(len(ret["ALGOS"])):
        str_body += f"& {ret['WAPE'][j]}"
    str_body += f"\\\\ \\cline{{2-{2+len(ret['ALGOS'])}}}\n"

    str_body += f"                      & SPEED   "
    for i in range(len(ret["ALGOS"])):
        str_body += f"& {ret['SPEED'][i]}"
    str_body += "\\\\ \\hline\n"

    return str_body

def gen_latex_table_code(report,out_dir,horizon):
    if isinstance(horizon,list):
        latex_code = latex_table_head_multi_col(report,horizon)
    else:
        latex_code = latex_table_head_single_col(report)
    
    for k,v in report.items():
        if isinstance(horizon,list):
            latex_code += latex_table_body_multi_col(k,v,horizon)
        else:
            latex_code += latex_table_body_single_col(k,v)
    latex_code += latex_table_tail()

    report_dir = f"{out_dir}/report"
    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)

    latex_code_path = f"{report_dir}/result_latex_table"
    with open(latex_code_path, 'w') as f:
        f.write(latex_code)

def gen_report_csv(report,out_dir):
    result = {k:[] for k in ["SOURCE","TARGET"] + list(report[list(report.keys())[0]].keys())}
    for k1,v1 in report.items():
        s,t = k1.split("|")
        result["SOURCE"] += [s]*len(v1["ALGOS"])
        result["TARGET"] += [t]*len(v1["ALGOS"])

        for k2,v2 in v1.items():
            if isinstance(v2[-1],list):
                # ret = np.array(v2).astype(float).mean(axis=0).tolist()
                # result[k2] += [str(f"{i:.2f}") for i in ret]
                result[k2] += v2[-1]
            else:
                result[k2] += v2

    report_df = pd.DataFrame(result)
    report_dir = f"{out_dir}/report"
    report_df.to_csv(f"{report_dir}/result.csv")
    return report_df

def gen_ranking_graph(report_df,out_dir,auto_open=False):
    fig = px.histogram(report_df, 
                        x="ALGOS", 
                        y=["SPEED","MAE","WAPE","MAPE"], 
                        width=800,
                        height=500,
                        color_discrete_map={
                            "SPEED": "RebeccaPurple", "MAE": "lightsalmon"
                            },
                        template="simple_white",
                        facet_col="SOURCE",
                        facet_col_wrap = 3,
                        histnorm = "percent"
                        )

    fig.update_layout(title="Efficiency analysis", 
                    font_family="San Serif",
                    bargap=0.2,
                    barmode='group',
                    titlefont={'size': 24},
                    legend=dict(
                    orientation="v", y=1, yanchor="top", x=1.25, xanchor="right")                
                    )

    # fig.show()
    report_dir = f"{out_dir}/report"
    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)
    offline.plot(fig, filename=f'{report_dir}/ranking.html', auto_open=auto_open)


def gen_polar_graph(report_df,out_dir,auto_open=False):
    fig = px.line_polar(report_df, r='SPEED', theta='ALGOS', color="SOURCE",line_close=True)

    # fig.show()
    report_dir = f"{out_dir}/report"
    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)
    offline.plot(fig, filename=f'{report_dir}/polar.html', auto_open=auto_open)
