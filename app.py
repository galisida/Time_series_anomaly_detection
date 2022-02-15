import requests
from flask import Flask, request, jsonify, abort
from train import *
import json
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return '<h1>Hello World!</h1>'


@app.route('/run', methods=['POST'])
def run():
    run_model(json.loads(request.get_data(as_text=True)))
    return jsonify({'msg': 'success', 'code': "200"})

# @app.route('/json', methods=['POST'])
# def json_():
#     data = json.loads(request.get_data(as_text=True))
#     print(data['code'])
#     print(data['msg'])
#     return data["code"], data['msg']


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)