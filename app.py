# save this as app.py
from flask import Flask

from tutorial_lastfm import embedding_from_string

app = Flask(__name__)

@app.get('/')
def hello():
    return 'Hello, World!'

@app.get('/recommend/<int:userid>')
def login_post(userid):
    return embedding_from_string(userid)
