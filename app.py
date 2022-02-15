from flask import Flask, request
from train import *
import json
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return '<h1>Hello World!</h1>'


@app.route('/run', methods=['POST'])
def run():
    res = run_model(json.loads(request.get_data(as_text=True)))
    # res = res.to_json(orient='records')
    res = res.to_dict(orient='dict')
    resp = dict()
    resp['data'] = res
    msg_json = {'msg': 'success', 'code': "200"}
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