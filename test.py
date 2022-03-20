from x3d_inference import ActionPredictionManager

ac = ActionPredictionManager(num_workers=1)

ac.put('videos/crops11/scooter_1.mp4')

ac.shutdown()