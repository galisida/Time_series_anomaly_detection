import numpy as np


def adjust_res_for_single(data, preload_csv, req_json, polution_id=None, meta=None):
    return_body = dict()
    if len(data) == 0:
        return None, None, None, None, None
    print('data_len: ', len(data))
    print('data: ', data)
    print('data[0]: ', data[0])
    print('data[0]的date: ', data[0]['date'])
    print('data[0]的loss: ', data[0]['loss'])
    print('data[0]的truth: ', data[0]['truth'])
    print('data[0]的test_result: ', data[0]['test_result'])

    print('data[0]的date的keys: ', data[0]['date'].keys())
    print('data[0]的loss的keys: ', data[0]['loss'].keys())
    print('data[0]的truth的keys: ', data[0]['truth'].keys())
    print('data[0]的test_result的keys: ', data[0]['test_result'].keys())
    keys_date1 = data[0]['date'].keys()
    keys_date1_len = len(keys_date1)

    keys_date2 = data[1]['date'].keys()
    keys_date2_len = len(keys_date2)

    date1_date_list = []
    date1_concentration_truth_list = []
    date1_concentration_test_list = []
    date1_concentration_loss_list = []
    date1_amount_truth_list = []
    date1_amount_test_list = []
    date1_amount_loss_list = []

    date2_date_list = []
    date2_concentration_truth_list = []
    date2_concentration_test_list = []
    date2_concentration_loss_list = []
    date2_amount_truth_list = []
    date2_amount_test_list = []
    date2_amount_loss_list = []

    for i in keys_date1:
        date1_date_list.append(data[0]['date'][i])
        date1_concentration_truth_list.append(data[0]['truth'][i])
        date1_concentration_test_list.append(data[0]['test_result'][i])
        date1_concentration_loss_list.append(data[0]['loss'][i])
        date1_amount_truth_list.append(data[2]['truth'][i])
        date1_amount_test_list.append(data[2]['test_result'][i])
        date1_amount_loss_list.append(data[2]['loss'][i])

    for i in keys_date2:
        date2_date_list.append(data[1]['date'][i])
        date2_concentration_truth_list.append(data[1]['truth'][i])
        date2_concentration_test_list.append(data[1]['test_result'][i])
        date2_concentration_loss_list.append(data[1]['loss'][i])
        date2_amount_truth_list.append(data[3]['truth'][i])
        date2_amount_test_list.append(data[3]['test_result'][i])
        date2_amount_loss_list.append(data[3]['loss'][i])

    # param
    return_body['date1'] = date1_date_list
    return_body['date2'] = date2_date_list

    date1_date_list = np.array(date1_date_list)
    date1_concentration_truth_list = np.array(date1_concentration_truth_list)
    date1_concentration_test_list = np.array(date1_concentration_test_list)
    date1_concentration_loss_list = np.array(date1_concentration_loss_list)
    date1_amount_truth_list = np.array(date1_amount_truth_list)
    date1_amount_test_list = np.array(date1_amount_test_list)
    date1_amount_loss_list = np.array(date1_amount_loss_list)

    date2_date_list = np.array(date2_date_list)
    date2_concentration_truth_list = np.array(date2_concentration_truth_list)
    date2_concentration_test_list = np.array(date2_concentration_test_list)
    date2_concentration_loss_list = np.array(date2_concentration_loss_list)
    date2_amount_truth_list = np.array(date2_amount_truth_list)
    date2_amount_test_list = np.array(date2_amount_test_list)
    date2_amount_loss_list = np.array(date2_amount_loss_list)

    # req_json = dict()
    # req_json['threshold_concentration_loss'] = 200.0
    # req_json['threshold_amount_loss'] = 2000.0
    # req_json['port_id'] = '857290898923237'
    # req_json['company_id'] = '172859950800371000'

    # param1 平均量
    average_amount = np.average(date1_amount_truth_list)
    print('average_amount: ', average_amount)
    return_body['average_amount'] = average_amount
    # param2 平均浓度
    average_concentration = np.average(date1_concentration_truth_list)
    print('average_concentration: ', average_concentration)
    return_body['average_concentration'] = average_concentration
    # 平均浓度损失
    average_concentration_loss = np.average(date1_concentration_loss_list)
    print('average_concentration_loss: ', average_concentration_loss)

    # param3 平均浓度损失较阈值
    date1_compare_concentration_loss_with_threshold = average_concentration_loss - req_json['threshold_concentration_loss']
    return_body['date1_compare_concentration_loss_with_threshold'] = date1_compare_concentration_loss_with_threshold

    # 平均量损失
    average_amount_loss = np.average(date1_amount_loss_list)
    print('average_amount_loss: ', average_amount_loss)
    # param4 ? 平均排放量量损失较阈值
    date1_compare_amount_loss_with_threshold = average_amount_loss - req_json['threshold_amount_loss']
    return_body['date1_compare_amount_loss_with_threshold'] = date1_compare_amount_loss_with_threshold




    compared_data_average_amount = np.average(date2_amount_truth_list)
    print('compared_data_average_amount: ', compared_data_average_amount)
    compared_data_average_concentration = np.average(date2_concentration_truth_list)
    print('compared_data_average_concentration: ', compared_data_average_concentration)
    compared_data_average_concentration_loss = np.average(date2_concentration_loss_list)
    print('compared_data_average_concentration_loss: ', compared_data_average_concentration_loss)
    compared_data_average_amount_loss = np.average(date2_amount_loss_list)
    print('compared_data_average_amount_loss: ', compared_data_average_amount_loss)

    # param5 较[上周(date2)]的average_concentration_loss(?)增长量(较上周浓度的损失增长量)
    data1_compare_data2_with_average_concentration_loss = average_concentration_loss - compared_data_average_concentration_loss
    return_body['data1_compare_data2_with_average_concentration_loss'] = data1_compare_data2_with_average_concentration_loss
    # param6 blablabla... 增长率
    data1_compare_data2_with_average_concentration_loss_rate = data1_compare_data2_with_average_concentration_loss / compared_data_average_concentration_loss
    return_body['data1_compare_data2_with_average_concentration_loss_rate'] = data1_compare_data2_with_average_concentration_loss_rate
    # param7 较[上周(date2)]的average_amount_loss(?)增长量(较上周排放量的损失增长量)
    data1_compare_data2_with_average_amount_loss = average_amount_loss - compared_data_average_amount_loss
    return_body['data1_compare_data2_with_average_amount_loss'] = data1_compare_data2_with_average_amount_loss
    # param8 blablabla... 增长率
    data1_compare_data2_with_average_amount_loss_rate = data1_compare_data2_with_average_amount_loss / compared_data_average_amount_loss
    return_body['data1_compare_data2_with_average_amount_loss_rate'] = data1_compare_data2_with_average_amount_loss_rate




    # print('-' * 30)
    # print(date2_amount_loss_list)
    # print('-' * 30)



    date1_concentration_loss_list_count = np.zeros_like(date1_concentration_loss_list)
    date1_amount_loss_list_count = np.zeros_like(date1_amount_loss_list)
    date2_concentration_loss_list_count = np.zeros_like(date2_concentration_loss_list)
    date2_amount_loss_list_count = np.zeros_like(date2_amount_loss_list)

    # 计算次数
    for i in range(len(date1_concentration_loss_list)):
        if abs(date1_concentration_loss_list[i]) > req_json['threshold_concentration_loss']:
            date1_concentration_loss_list_count[i] = 1
        else:
            date1_concentration_loss_list_count[i] = 0

    for i in range(len(date1_amount_loss_list)):
        if abs(date1_amount_loss_list[i]) > req_json['threshold_amount_loss']:
            date1_amount_loss_list_count[i] = 1
        else:
            date1_amount_loss_list_count[i] = 0

    for i in range(len(date2_concentration_loss_list)):
        if abs(date2_concentration_loss_list[i]) > req_json['threshold_concentration_loss']:
            date2_concentration_loss_list_count[i] = 1
        else:
            date2_concentration_loss_list_count[i] = 0

    for i in range(len(date2_amount_loss_list)):
        if abs(date2_amount_loss_list[i]) > req_json['threshold_amount_loss']:
            date2_amount_loss_list_count[i] = 1
        else:
            date2_amount_loss_list_count[i] = 0


    # 计算较阈值
    # date1_sum_loss_list = date1_concentration_loss_list + date1_amount_loss_list
    # date2_sum_loss_list = date2_concentration_loss_list + date2_amount_loss_list

    # 计算较阈值，太大就预警
    # 较阈值用一个average loss表示好了
    date1_compare_concentration_diff_list = np.abs(date1_concentration_loss_list) - np.abs(req_json['threshold_concentration_loss'])
    date1_compare_amount_diff_list = np.abs(date1_amount_loss_list) - np.abs(req_json['threshold_amount_loss'])
    date2_compare_concentration_diff_list = np.abs(date2_concentration_loss_list) - np.abs(req_json['threshold_concentration_loss'])
    date2_compare_amount_diff_list = np.abs(date2_amount_loss_list) - np.abs(req_json['threshold_amount_loss'])

    print('date1_concentration_loss_list: ', date1_concentration_loss_list)
    print('date1_amount_loss_list: ', date1_amount_loss_list)
    print('date2_concentration_loss_list: ', date2_concentration_loss_list)
    print('date2_amount_loss_list: ', date2_amount_loss_list)

    print('date1_compare_concentration_diff_list: ', date1_compare_concentration_diff_list)
    print('date1_compare_amount_diff_list: ', date1_compare_amount_diff_list)
    print('date2_compare_concentration_diff_list: ', date2_compare_concentration_diff_list)
    print('date2_compare_amount_diff_list: ', date2_compare_amount_diff_list)

    # 预警日期
    date1_concentration_warning_tags = [date1_date_list[i] for i in range(len(date1_compare_concentration_diff_list)) if date1_compare_concentration_diff_list[i] > 0]
    date1_amount_warning_tags = [date1_date_list[i] for i in range(len(date1_compare_amount_diff_list)) if date1_compare_amount_diff_list[i] > 0]
    date2_concentration_warning_tags = [date2_date_list[i] for i in range(len(date2_compare_concentration_diff_list)) if date2_compare_concentration_diff_list[i] > 0]
    date2_amount_warning_tags = [date2_date_list[i] for i in range(len(date2_compare_amount_diff_list)) if date2_compare_amount_diff_list[i] > 0]

    print('date1_concentration_warning_tags: ', date1_concentration_warning_tags)
    print('date1_amount_warning_tags: ', date1_amount_warning_tags)
    print('date2_concentration_warning_tags: ', date2_concentration_warning_tags)
    print('date2_amount_warning_tags: ', date2_amount_warning_tags)

    # param9 当前检测时间段内疑似违规天数 浓度异常
    date1_concentration_warning_tags_count = len(date1_concentration_warning_tags)
    return_body['date1_concentration_warning_tags_count'] = date1_concentration_warning_tags_count
    # param10 浓度异常具体日期  date1_concentration_warning_tags
    return_body['date1_concentration_warning_tags'] = date1_concentration_warning_tags
    # param11 ? 当前检测时间段内疑似违规天数 排放量异常
    date1_amount_warning_tags_count = len(date1_amount_warning_tags)
    return_body['date1_amount_warning_tags_count'] = date1_amount_warning_tags_count
    # param12 排放量异常具体日期  date1_amount_warning_tags
    return_body['date1_amount_warning_tags'] = date1_amount_warning_tags

    # part2:
    # param13 [上周平均浓度]
    date2_average_concentration_truth = np.average(date2_concentration_truth_list)
    print('date2_average_concentration_truth: ', date2_average_concentration_truth)
    return_body['date2_average_concentration_truth'] = date2_average_concentration_truth
    # param14 [上周平均排放量]
    date2_average_amount_truth = np.average(date2_amount_truth_list)
    print('date2_average_amount_truth: ', date2_average_amount_truth)
    return_body['date2_average_amount_truth'] = date2_average_amount_truth

    # 可视化
    # param15 当前检测时间段的浓度
    # date1_concentration_truth_list
    return_body['date1_concentration_truth_list'] = date1_concentration_truth_list.tolist()
    # param16 当前检测时间段的经过模型重构的浓度
    # date1_concentration_test_list
    return_body['date1_concentration_test_list'] = date1_concentration_test_list.tolist()

    # param17 当前检测时间段的排放量
    # date1_amount_truth_list
    return_body['date1_amount_truth_list'] = date1_amount_truth_list.tolist()
    # param18 当前检测时间段的经过模型重构的排放量
    # date1_amount_test_list
    return_body['date1_amount_test_list'] = date1_amount_test_list.tolist()

    # param19 [上周]的浓度
    # date2_concentration_truth_list
    return_body['date2_concentration_truth_list'] = date2_concentration_truth_list.tolist()
    # param20 [上周]的浓度
    # date2_amount_truth_list
    return_body['date2_amount_truth_list'] = date2_amount_truth_list.tolist()

    # 关联性:
    port_id = req_json['port_id']

    port_relative_df = preload_csv[preload_csv['port_id'].str.contains(port_id)]
    print('port_relative_df:')
    print(port_relative_df)
    # param21 排污口的排污总量
    port_total_concentration = np.sum(port_relative_df['concentration'].values)
    print(port_total_concentration)
    return_body['port_total_concentration'] = port_total_concentration
    # param22 排污口平均排污浓度
    port_average_amount = np.average(port_relative_df['amount'].values)
    print(port_average_amount)
    return_body['port_average_amount'] = port_average_amount
    # param23 其他共用企业id
    relative_cpn = port_relative_df['company_id'].values.astype('str')
    relative_cpn = np.unique(relative_cpn)
    # print(req_json['company_id'])
    relative_cpn = [cpn for cpn in relative_cpn if cpn.strip() != req_json['company_id']]
    print(relative_cpn)
    return_body['relative_cpn'] = relative_cpn

    # param24 污染物id
    return_body['polution_id'] = req_json['polution_id']
    if return_body['polution_id'] == '':
        return_body['polution_id'] = polution_id


    return return_body, date1_concentration_warning_tags_count, date1_amount_warning_tags_count, \
           len(date2_concentration_warning_tags), len(date2_amount_warning_tags)
