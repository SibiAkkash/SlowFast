from time import time
from flask import Flask, request
from x3d_inference import ActionPredictionManager
import json
import time

app = Flask(__name__)

acm = ActionPredictionManager()


@app.route("/hello", methods=["GET"])
def send_hello():
    return "hello\n"
    

@app.route("/api/action_recog", methods=["POST"])
def action_recog():
    print(request)
    print(type(request.json))
    time.sleep(2)

    data = json.loads(request.json)
    print(data, type(data))

    video_path = data["video_path"]
    print(video_path)

    acm.put(video_path)

    return {"msg": "received video path"}


if __name__ == "__main__":
    app.run(host="10.121.9.138", port=1233)
