from flask import Flask
from flask import request
from flask import jsonify

import numpy as np
from keras.models import load_model
import keras_preprocessing.image

import base64
from io import BytesIO
from PIL import Image
import requests

app = Flask(__name__)

model = None
dictionary = ['axes', 'boots', 'carabiners', 'crampons', 'gloves', 'hardshell_jackets', 'harnesses', 'helmets', 'insulated_jackets', 'pulleys', 'rope', 'tents']

@app.route("/", methods=['GET', 'POST'])
def index():
  global model, dictionary
  if request.method == 'POST':
    if model == None:
      model = load_model('cnn_model_gears.h5')

    response = requests.get(request.data)
    imageByte = BytesIO(response.content)

    image = keras_preprocessing.image.load_img(imageByte, target_size=(128,128))
    # buffered = BytesIO()
    # image.save(buffered, format="JPEG")
    imageData = base64.b64encode(imageByte.getvalue())
    # print(np.array(image).shape)
    return jsonify(
      imageData=str(imageData)[2:-1],
      prediction=dictionary[model.predict(np.array([np.array(image)])).argmax()]
    )
  else:
    return """<style>body{text-align:center;}input{width:70%;}img{width:300px;}select{display:inline-block}</style>
    <input/><button id="predict">Predict</button>
    <div>
      <select>
        <option></option>
        <option value="https://images-na.ssl-images-amazon.com/images/I/71WDJfk%2BD3L._SL1500_.jpg">axe</option>
        <option value="https://i5.walmartimages.com/asr/9ad142d8-09c5-43c9-a618-646f75701d38_1.aca00e67749f0075fa12c4ee2f488510.jpeg">boot</option>
        <option value="http://www.blackdiamondequipment.com/on/demandware.static/-/Sites-bdel/default/dwaad1abaf/products/carabiners_draws/S16/210275_RockLock_Screwgate_Carabiner_web.jpg">carabiner</option>
        <option value="https://contents.mediadecathlon.com/p341063/2000x2000/sq/monoceros_speed_crampons_simond_8320468_341063.jpg?k=6bb5e91ef2a72cf2fd0a024097ac9dc1">crampon</option>
        <option value="http://www.wigglestatic.com/product-media/5360107487/SealSkinz-All-Weather-XP-Cycle-Gloves-Long-Finger-Gloves-Hi-Vis-Yellow-Black-AW15-121150870110-3.jpg">gloves</option>
        <option value="https://outdoorgearlab-mvnab3pwrvp3t0.stackpathdns.com/photos/16/51/286670_31243_XXL.jpg">hardshell jacket</option>
        <option value="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQHlj23wH2OGTo44tQf89tPENY8A9Cn9_hLfsgbiVaFBrOexjnT">harness</option>
        <option value="https://shop.epictv.com/sites/default/files/ae42ad29e70ba8ce6b67d3bdb6ab5c6e.jpeg">helmet</option>
        <option value="https://image.sportsmansguide.com/adimgs/l/6/673942i3_ts.jpg">insulated jacket</option>
        <option value="https://images.homedepot-static.com/productImages/52e2c992-1f7e-4f08-b093-a5b4e17f2445/svn/everbilt-rope-chain-connectors-43344-64_1000.jpg">pulleys(tent)</option>
        <option value="https://images.homedepot-static.com/productImages/7379473d-9c2c-4850-aebc-3e8ef6d78a99/svn/everbilt-rope-chain-connectors-42604-64_1000.jpg">pulleys(rope)</option>
        <option value="https://www.bluewaterropes.com/wp-content/uploads/2013/11/BWR3-YEBK.jpg">rope</option>
        <option value="https://cdn.shopify.com/s/files/1/1832/7001/products/teepee-tent-2-1500.jpg?v=1491604775">tent</option>
      </select>
    </div>
    <div id="status"></div>
    <div><img><div id="prediction"></div></div>
    <script>
      predict=()=>{
        document.querySelector('#status').textContent = "predicting..."
        fetch('/',{method: 'POST', body:document.querySelector('input').value})
        .then((res)=>{
          window.res=res;
          res.json()
          .then(json=>{
            document.querySelector('#prediction').textContent = json.prediction;
            document.querySelector('img').src=`data:image/png;base64,${json.imageData}`
          })
        })
        .finally(()=>{document.querySelector('#status').textContent = ""})
      };
      select=()=>{
        document.querySelector('input').value = document.querySelector('select').value;
      }
      document.getElementById('predict').addEventListener('click',predict)
      document.querySelector('select').addEventListener('change',select)
    </script>"""
