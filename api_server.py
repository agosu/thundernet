from flask_ngrok import run_with_ngrok
from flask import Flask
from flask import request

from training import get_api_result

app = Flask(__name__)
run_with_ngrok(app)  # starts ngrok when the app is run


@app.route("/")
def home():
    return "<h1>Running Flask on Google Colab!</h1>"


@app.route("/detect")
def detect():
    img_path = request.args.get('path')
    return get_api_result(img_path)


app.run()
