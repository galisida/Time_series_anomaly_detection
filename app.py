from flask import Flask, request
from train import *
import json
from flask_cors import CORS
from pandas import read_csv
from utils.adjust_res import *

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def index():
    return '<h1>Hello World!</h1>'


preloaded_csv = read_csv('dataset/all_data.csv')
preloaded_csv = preloaded_csv.fillna({'concentration': 0, 'amount': 0})


@app.route('/run', methods=['POST'])
def run():
    req_json = json.loads(request.get_data(as_text=True))
    res, meta, polution_ids = run_model(req_json, preloaded_csv)
    # res = res.to_json(orient='records')
    # res = res.to_dict(orient='dict')
    if meta == "single":
        res = [r.to_dict() for r in res]
    else:
        for i in range(len(res)):
            for j in range(len(res[i])):
                res[i][j] = res[i][j].to_dict()

    # param: 一些总览违规次数
    date1_total_concentration_warning_count = 0
    date1_total_amount_warning_count = 0
    date2_total_concentration_warning_count = 0
    date2_total_amount_warning_count = 0

    # todo: adjust res check here
    if meta == 'single':
        res, _, _, _, _ = adjust_res_for_single(res, preloaded_csv, req_json)
    elif meta == 'all':
        # todo: adjust res for all polutions
        for polution_idx in range(len(res)):
            res[polution_idx], date1_con_cnt, date1_amount_cnt, \
            date2_con_cnt, date2_amount_cnt = adjust_res_for_single(res[polution_idx], preloaded_csv, req_json,
                                                                    polution_ids[polution_idx], meta)
            date1_total_concentration_warning_count += date1_con_cnt
            date1_total_amount_warning_count += date1_amount_cnt
            date2_total_concentration_warning_count += date2_con_cnt
            date2_total_amount_warning_count += date2_amount_cnt

    # param
    # [本周] 总览浓度异常次数
    overview_dict = dict()
    overview_dict['date1_total_concentration_warning_count'] = date1_total_concentration_warning_count
    # [本周] 总览排放量异常次数
    overview_dict['date1_total_amount_warning_count'] = date1_total_amount_warning_count
    # [本周] 总览浓度与排放量异常总次数
    overview_dict['date1_total_warning_count'] = date1_total_concentration_warning_count + date1_total_amount_warning_count
    # [上周] 总览浓度异常次数
    overview_dict['date2_total_concentration_warning_count'] = date2_total_concentration_warning_count
    # [上周] 总览排放量异常次数
    overview_dict['date2_total_amount_warning_count'] = date2_total_amount_warning_count
    # [上周] 总览浓度与排放量异常总次数
    overview_dict['date2_total_warning_count'] = date2_total_concentration_warning_count + date2_total_amount_warning_count

    # 较[上周] 总览浓度异常次数增加
    overview_dict['compare_total_concentration_warning_count'] = date1_total_concentration_warning_count - date2_total_concentration_warning_count
    # 较[上周] 总览浓度异常次数增加率
    if date2_total_concentration_warning_count == 0:
        overview_dict['compare_total_concentration_warning_count_rate'] = -1
    else:
        overview_dict['compare_total_concentration_warning_count_rate'] = (date1_total_concentration_warning_count - date2_total_concentration_warning_count) / date2_total_concentration_warning_count
    # 较[上周] 总览排放量异常次数增加
    overview_dict['compare_total_amount_warning_count'] = date1_total_amount_warning_count - date2_total_amount_warning_count
    # 较[上周] 总览排放量异常次数增加率
    if date2_total_amount_warning_count == 0:
        overview_dict['compare_total_amount_warning_count_rate'] = -1
    else:
        overview_dict['compare_total_amount_warning_count_rate'] = (date1_total_amount_warning_count - date2_total_amount_warning_count) / date2_total_amount_warning_count
    # 较[上周] 总览浓度与排放量异常总次数增加
    overview_dict['compare_total_warning_count'] = overview_dict['compare_total_concentration_warning_count'] + overview_dict['compare_total_amount_warning_count']
    # 较[上周] 总览浓度与排放量异常总次数增加率
    flag = 0
    if overview_dict['compare_total_concentration_warning_count_rate'] == -1:
        flag += 1
    if overview_dict['compare_total_amount_warning_count_rate'] == -1:
        flag += 1
    overview_dict['compare_total_warning_count_rate'] = flag + overview_dict['compare_total_concentration_warning_count_rate'] + overview_dict['compare_total_amount_warning_count_rate']




    resp = dict()
    resp['data'] = res
    resp['overview'] = overview_dict
    msg_json = {'msg': 'success', 'code': "200", 'meta': meta}
    resp.update(msg_json)
    return resp


# @app.route('/json', methods=['POST'])
# def json_():
#     data = json.loads(request.get_data(as_text=True))
#     print(data['code'])
#     print(data['msg'])
#     dic1 = {'msg': 'success', 'code': "200"}
#     dic2 = {'msg2': 'success2', 'code2': "2002"}
#     resp = dic1.update(dic2) # 返回的是None
#     print(dic1)
#     return dic1


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)
