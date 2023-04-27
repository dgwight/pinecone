# save this as app.py
from flask import Flask

from tutorial_lastfm import (recommend_for, get_similar)

app = Flask(__name__)

@app.get('/')
def hello():
    return 'Hello, World!'

@app.get('/recommend/<int:userid>')
def recommend(userid):
    return recommend_for(userid)

@app.get('/similar/<int:itemid>')
def similar(itemid):
    return get_similar(itemid)
