* Run the `server.py` file, set appropriate host and port. This starts the server that listens to incoming scooter video paths. 
* When the server receives a video path, it enqueues it for action prediction. The action prediction model is loaded and this video is sent for inference.

* Change the `DIR_ROOT`, `INPUT_VIDEO_DIR`, `OUTPUT_VIDEO_DIR`, as appropirate.
* The input path to the api endpoint is relative to `INPUT_VIDEO_DIR`
* The output video annotated with action predictions is saved to `OUTPUT_VIDEO_DIR`
