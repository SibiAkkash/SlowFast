from flask import Flask, request
from x3d_inference import ActionPredictionManager
import json

app = Flask(__name__)

acm = ActionPredictionManager()

@app.route("/api/action_recog", methods=["POST"])
def action_recog():
    print(request)
    print(type(request.json))
    
    data = json.loads(request.json)
    print(data, type(data))
    
    video_path = data["video_path"]
    print(video_path)

    acm.put(video_path)

    return {"msg": "received video path"}
    