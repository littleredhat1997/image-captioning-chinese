#-*- coding:utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import caption
import time
import hashlib
import numpy as np

from flask import Flask, request, make_response
app = Flask(__name__)

ERROR = "sorry......"
ALLOWED_EXTENSIONS = set(['jpg', 'png'])
UPLOAD_FOLDER = './upload'


def predict(path):
    t = time.time()
    result = caption.load(path)

    print("Mission complete, it took {:.3f}s".format(time.time() - t))
    print("Recognition Result:", result)
    
    response = make_response(result)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'        
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/api', methods=['GET', 'POST'])
def caption_api():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            content = file.read()
            path = os.path.join(UPLOAD_FOLDER, hashlib.md5(content).hexdigest())
            image = open(path, 'wb')
            image.write(content)
            return predict(path)
    if request.method == 'GET':
        id = int(request.args.get('id'))
        if id >= 5 and id <= 8:
            path = '/home/ubuntu/桌面/test/images/00%d.jpg' % id
            return predict(path)
    return ERROR


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
