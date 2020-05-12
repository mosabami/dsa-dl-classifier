#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 18:17:24 2018

@author: vivekkalyanarangan
"""

# For Genrating test images
#from PIL import Image
#from keras.datasets import mnist
#import numpy as np
#
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#for i in np.random.randint(0, 10000+1, 10):
#    arr2im = Image.fromarray(X_train[i])
#    arr2im.save('{}.png'.format(i), "PNG")

import keras
from keras.models import load_model,model_from_json
from PIL import Image
import numpy as np
from flasgger import Swagger

from flask import Flask, request
app = Flask(__name__)
swagger = Swagger(app)
keras.backend.clear_session()


# A welcome message to test our server
@app.route('/')
    return """<h1>Welcome to our deep learning classification webapp !!</h1> 
    <p>go to <a href="/apidocs">app page<a> to access the app<p>"""

@app.route('/makepredict', methods=['POST'])
def predict_digit():
    """
    Example endpoint returning a prediction of mnist
    ---
    parameters:
      - name: image
        in: formData
        type: file
        required: true
    definitions:
      value:
        type: object
        properties:
          value_name:
            type: string
            items:
              $ref: '#/definitions/Color'
      Color:
        type: string
    responses:
      200:
        description: OK
        schema:
          $ref: '#/definitions/value'
    """
    im = Image.open(request.files['image'])
    im2arr = np.array(im).reshape((1, 1, 28, 28))
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return str(np.argmax(loaded_model.predict(im2arr)))

if __name__ == '__main__':
    app.run(threaded=True, port=5000)


#im = Image.open("8432.png")
#im2arr = np.array(im).reshape((1, 1, 28, 28))
#model2 = model.make_predict_function()
#str(np.argmax(model2.predict(im2arr)))
