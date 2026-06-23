"""
detection package

Public API:  detect(), detect_file()  from detection.detect

Pose keypoint ordering (COCO 17-point skeleton, used in DetectionRecord.pose_keypoints):
  Index  Name
  -----  ----
  0      nose
  1      left_eye
  2      right_eye
  3      left_ear
  4      right_ear
  5      left_shoulder   <- seatbelt / helmet check
  6      right_shoulder  <- seatbelt / helmet check
  7      left_elbow
  8      right_elbow
  9      left_wrist
  10     right_wrist
  11     left_hip
  12     right_hip
  13     left_knee
  14     right_knee
  15     left_ankle
  16     right_ankle

Head region for helmet check: keypoints 0-4 (nose + ears + eyes).
Torso region for seatbelt check: keypoints 5-6 (shoulders) and 11-12 (hips).
"""
