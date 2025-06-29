import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque

# --- Configurations ---
MODEL_PATH = "D:/Vs Code/Player Reidentification using opencv/best.pt"
VIDEO_PATH = "D:/Vs Code/Player Reidentification using opencv/15sec_input_720p.mp4"
OUTPUT_VIDEO_PATH = "output_reidentified_video_2.mp4"

# --- Check Files ---
assert os.path.exists(MODEL_PATH), "Model file not found."
assert os.path.exists(VIDEO_PATH), "Video file not found."

# --- Load Model ---
model = YOLO(MODEL_PATH)

# --- Parameters ---
MAX_DIST_ACTIVE = 70
MAX_LOST = 7
REID_DIST = 200
REID_TOLERANCE = 0.3
REID_MAX_LOST = 75
WEIGHT_DIST = 0.6

# --- Helpers ---
def get_centroid(bbox):
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter_area / (areaA + areaB - inter_area + 1e-6)

def bbox_similarity(a, b):
    wa, ha = a[2] - a[0], a[3] - a[1]
    wb, hb = b[2] - b[0], b[3] - b[1]
    return (min(wa, wb) / max(wa, wb) + min(ha, hb) / max(ha, hb)) / 2

def match_score(det_bbox, det_centroid, track, match_type):
    tb = track['last_bbox']
    tc = track['last_centroid']
    dist = np.linalg.norm(np.subtract(det_centroid, tc)) / (REID_DIST if match_type == 'reid' else MAX_DIST_ACTIVE)
    sim = 1 - bbox_similarity(det_bbox, tb)
    iou_score = iou(det_bbox, tb)
    if match_type == 'active':
        return min(dist * WEIGHT_DIST + (1 - iou_score) * (1 - WEIGHT_DIST), 1.5) if iou_score > 0.01 else dist * 2
    return dist * WEIGHT_DIST + sim * (1 - WEIGHT_DIST) + (0.5 if iou_score < 0.1 else 0)

def get_color(pid):
    np.random.seed(pid)
    return tuple(np.random.randint(100, 255, 3).tolist())

# --- Video Setup ---
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# --- Tracking Initialization ---
active_tracks = {}
lost_tracks = {}
next_id = 0

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5, iou=0.7, verbose=False)
    detections = []
    for *xyxy, conf, cls in results[0].boxes.data.cpu().numpy():
        if model.names[int(cls)] == 'player':
            detections.append({'bbox': [int(x) for x in xyxy], 'confidence': float(conf)})

    assigned = set()
    current_ids = []

    # --- Active Matching ---
    matches = []
    for i, d in enumerate(detections):
        dc = get_centroid(d['bbox'])
        best, score = None, float('inf')
        for tid, t in active_tracks.items():
            s = match_score(d['bbox'], dc, t, 'active')
            if s < score and iou(d['bbox'], t['last_bbox']) > 0.05:
                best, score = tid, s
        if best is not None and score < 1.0:
            matches.append((i, best, score))

    for i, tid, _ in sorted(matches, key=lambda x: x[2]):
        if i in assigned or tid in current_ids:
            continue
        d = detections[i]
        d['player_id'] = tid
        active_tracks[tid].update({
            'last_bbox': d['bbox'],
            'last_centroid': get_centroid(d['bbox']),
            'frames_since_last_seen': 0,
            'last_frame_id': frame_idx
        })
        active_tracks[tid]['history'].append(d['bbox'])
        current_ids.append(tid)
        assigned.add(i)

    # --- Re-ID Matching ---
    rem = [(i, d) for i, d in enumerate(detections) if i not in assigned]
    reid_matches = []
    for tid, t in list(lost_tracks.items()):
        if frame_idx - t['first_lost_frame_id'] > REID_MAX_LOST:
            del lost_tracks[tid]
            continue
        best, score = None, float('inf')
        for i, d in rem:
            dc = get_centroid(d['bbox'])
            if np.linalg.norm(np.subtract(dc, t['last_centroid'])) > REID_DIST:
                continue
            sim = bbox_similarity(d['bbox'], t['last_bbox'])
            if sim < 1 - REID_TOLERANCE:
                continue
            s = match_score(d['bbox'], dc, t, 'reid')
            if s < score:
                best, score = i, s
        if best is not None and score < 1.0:
            reid_matches.append((best, tid, score))

    for i, tid, _ in sorted(reid_matches, key=lambda x: x[2]):
        if i in assigned:
            continue
        d = detections[i]
        d['player_id'] = tid
        active_tracks[tid] = {
            'last_bbox': d['bbox'],
            'last_centroid': get_centroid(d['bbox']),
            'frames_since_last_seen': 0,
            'history': deque([d['bbox']], maxlen=10),
            'last_frame_id': frame_idx
        }
        del lost_tracks[tid]
        assigned.add(i)

    # --- New Detections ---
    for i, d in enumerate(detections):
        if i in assigned:
            continue
        d['player_id'] = next_id
        active_tracks[next_id] = {
            'last_bbox': d['bbox'],
            'last_centroid': get_centroid(d['bbox']),
            'frames_since_last_seen': 0,
            'history': deque([d['bbox']], maxlen=10),
            'last_frame_id': frame_idx
        }
        next_id += 1

    # --- Track Management ---
    for tid in list(active_tracks.keys()):
        if not any(d.get('player_id') == tid for d in detections):
            active_tracks[tid]['frames_since_last_seen'] += 1
            if active_tracks[tid]['frames_since_last_seen'] > MAX_LOST:
                lost_tracks[tid] = {
                    'last_bbox': active_tracks[tid]['last_bbox'],
                    'last_centroid': active_tracks[tid]['last_centroid'],
                    'first_lost_frame_id': frame_idx,
                    'history_bboxes': list(active_tracks[tid]['history']),
                    'last_frame_id': frame_idx
                }
                del active_tracks[tid]

    # --- Draw and Display ---
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        pid = d['player_id']
        cv2.rectangle(frame, (x1, y1), (x2, y2), get_color(pid), 2)
        cv2.putText(frame, f"ID: {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Live Re-Identification", frame)
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Playback interrupted by user.")
        break

    frame_idx += 1
    if frame_idx % 25 == 0:
        print(f"Processed {frame_idx} frames.")

# --- Cleanup ---
cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")
