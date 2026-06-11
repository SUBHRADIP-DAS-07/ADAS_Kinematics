"""
ADAS Pipeline — modules package

Exposes the core pipeline components:
  data_loader     — Module 1: ingestion & sync
  detector        — Module 2: YOLOv8 vehicle detection
  depth_estimator — Module 3: depth / 3D position
  tracker         — Module 4: DeepSORT / IoU tracking
  coord_transform — Module 5: camera <-> CARLA world transform
  kalman_filter   — Module 6: state estimation
  predictor       — Module 7: trajectory prediction (CV/CA/LSTM)
  validator       — Module 8: ADE/FDE/RMSE validation
"""
