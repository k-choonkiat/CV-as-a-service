
import tensorflow_hub as hub
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
import numpy as np
import time
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file,jsonify
import io
import requests
import json
import base64
from methods import *

app = Flask(__name__)

def serve_pil_image(pil_img):

    img_io = io.BytesIO()
    pil_img.save(img_io, 'jpeg', quality=100)
    img_io.seek(0)
    img = base64.b64encode(img_io.getvalue()).decode()
    img_tag = f'<img src="data:image/jpg;base64,{img}" class="img-fluid"/>'
    return img_tag


def detect_img(image_url):
  start_time = time.time()
  image_path = download_and_resize_image(image_url, 640, 480)
  res_img = run_detector(detector, image_path)
  end_time = time.time()
  time_taken = end_time - start_time
  print("Inference time:",time_taken)
  return res_img,time_taken

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=['POST'])
def generate():
     path = request.form['imgName']
    
     image,time_taken = detect_img(path)

     img_tag=serve_pil_image(image)

    

     # return the data dictionary as a JSON response
     return render_template("index.html",picture=img_tag,time_taken=time_taken)



if __name__ == "__main__":
    module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]

    detector = hub.load(module_handle).signatures['default']
    
    app.run(debug=True)
