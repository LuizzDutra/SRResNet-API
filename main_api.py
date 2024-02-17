import io
import json
import os
import glob
import super_resolve as inference
from PIL import Image 


from flask import Flask, jsonify, request, make_response


app = Flask(__name__)


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if request.files is not None:
            image_list = []
            for f in request.files:
                #read image file string data
                filestr = request.files[f].read()

                img = Image.open(io.BytesIO(filestr))

                image_list.append((f, img))

            result = inference.visualize_sr(image_list[0][1])
        
            img_io = io.BytesIO()
            result.save(img_io, 'PNG')
            img_io.seek(0)

            response = make_response(img_io)
            response.headers.set('Content-Type', 'image/png')

            return response
        
        else:
            return {'msg': "not received"}

@app.route('/test', methods=['GET'])
def test():
    if request.method == 'GET':
        print("getted")
        return {'msg': "received"}

if __name__ == '__main__':
    app.run()
