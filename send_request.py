import requests
import json

url = "http://localhost:9999/api/action_recog"

headers = {"Content-Type": "application/json"}

data = {
    "video_path": "/media/sibi/DATA/dev/ai/internship/yolov5-train/videos/crops44/scooter_1.mp4"
}

resp = requests.post(url, headers=headers, json=json.dumps(data))
print(resp.status_code, resp.text)
