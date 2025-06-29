# Player Re-Identification in Sports Footage

This project implements **player re-identification in a single camera feed**.

## 🎯 Objective

Detect and consistently track football players in a **single 15-second match video**, such that:
- Each player is assigned a unique ID upon first appearance.
- If a player goes out of frame and returns later, they are re-identified with the **same ID**.

## 📁 Files

- `main_2_final.py` — Final optimized pipeline (single-pass processing, real-time display, and video saving).
- `15sec_input_720p.mp4` — Input video used for inference.
- `output_reidentified_video_2.mp4` — Output video with tracked and labeled players.


## ⚙️ Methodology

- **YOLOv11** detects players in each frame.
- **Active tracking** matches current frame detections with existing IDs using:
  - Centroid distance
  - IoU
- **Re-identification** is performed on lost tracks using:
  - Bounding box similarity
  - Distance-based heuristics
- **Single-pass processing** ensures real-time display and recording are handled efficiently.

## ⚠️ Limitations

- Real-time display may be slower due to per-frame inference.
- Appearance-based ID switching may occur during heavy occlusion.
- OCR jersey number detection was tested but not used in final version due to latency.

## 🚀 Future Improvements

- Integrate appearance-based deep re-ID (e.g., OSNet).
- Use trackers like Deep SORT or ByteTrack for improved association.
- Optimize inference for GPU to achieve true real-time FPS.

## 👨‍💻 Author

Akshay Bamnote  
B.Tech AI & ML  
MPSTME, NMIMS Shirpur  
