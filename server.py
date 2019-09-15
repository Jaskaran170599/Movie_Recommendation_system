from flask import Flask,request
from recommender import *
import json

app= Flask(__name__)

@app.route('/recommend/title/<full_id>',methods=["GET","POST"]) #title
def predict(full_id):
    movies=get_nearest_movie(full_id)
    res={
        "movies":str(movies)
    }
    res=json.dumps(res)
    return res

@app.route('/recommend/overview/<full_id>',methods=["GET","POST"]) #overview
def predict_overview(full_id):
    movies=get_nearest_overview(full_id)
    res={
        "movies":str(movies)
    }
    res=json.dumps(res)
    return res

if __name__=="__main__":
    app.run(port=3000,debug=True)