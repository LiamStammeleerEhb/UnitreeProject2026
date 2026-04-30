import asyncio
import json
import time
import logging
import threading
from pathlib import Path
from queue import Queue
import rclpy
from std_msgs.msg import String

import cv2
import cv2.aruco as aruco
import numpy as np
from ultralytics import YOLO
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack

# --- Settings ---
MODEL_PATH        = "/home/jetson/Models/kaai.pt"
DETECTION_CONF    = 0.3
SCAN_HEIGHTS      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
RECORDS_DIR       = Path("records")
SAVE_INTERVAL_SEC = 10.0
ROS_TOPIC         = "unitree_key_cmd"
SHOW_PREVIEW      = False
PRINT_EVERY_N_FRAMES = 5

# --- ArUco & Camera Settings ---
MARKER_SIZE = 0.068  # 6.8cm in meters
CAMERA_MATRIX = np.array([
    [1430, 0, 960],
    [0, 1430, 540],
    [0, 0, 1]
], dtype=np.float32)
DIST_COEFFS = np.zeros((5, 1))

# Initialize ArUco objects
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# 3D points for pose estimation
marker_points = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
], dtype=np.float32)

logging.basicConfig(level=logging.FATAL)
model = YOLO(MODEL_PATH, verbose=False)

# ── Processing Logic ─────────────────────────────────────────────────────────

def process_frame(frame: np.ndarray) -> tuple[np.ndarray, float, list]:
    """Run YOLO and ArUco detection on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # 1. YOLO Segmentation
    results = model(frame, conf=DETECTION_CONF, verbose=False)
    midpoints = []
    for r in results:
        if r.masks is not None and len(r.masks.data) > 0:
            mask = r.masks.data[0].cpu().numpy()
            mask = cv2.resize((mask * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            green = np.full_like(frame, (0, 255, 0))
            overlay[mask > 0] = cv2.addWeighted(frame, 0.3, green, 0.7, 0)[mask > 0]
            for rr in SCAN_HEIGHTS:
                y = int(h * rr)
                idx = np.where(mask[y, :] > 0)[0]
                if len(idx) > 0:
                    mx = int(np.mean(idx))
                    midpoints.append((mx, y))
                    cv2.circle(overlay, (mx, y), 5, (255, 0, 0), -1)

    # 2. ArUco Detection & Distance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    detected_markers = []

    if ids is not None:
        aruco.drawDetectedMarkers(overlay, corners, ids)
        for i in range(len(ids)):
            _, rvec, tvec = cv2.solvePnP(marker_points, corners[i], CAMERA_MATRIX, DIST_COEFFS)
            dist = float(np.linalg.norm(tvec))
            marker_id = int(ids[i][0])
            detected_markers.append({"id": marker_id, "distance": round(dist, 3)})
            
            # Draw distance on overlay
            c = corners[i].reshape(4, 2)
            cv2.putText(overlay, f"ID:{marker_id} {dist:.2f}m", (int(c[0][0]), int(c[0][1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 3. Heading Calculation
    start_point = (w // 2, h)
    direction_angle = 90.0
    if midpoints:
        avg_x = int(np.mean([p[0] for p in midpoints]))
        target_point = (avg_x, min(p[1] for p in midpoints))
        direction_angle = float(np.degrees(np.arctan2(start_point[1] - target_point[1], avg_x - start_point[0])))
        cv2.arrowedLine(overlay, start_point, target_point, (0, 0, 255), 5)

    return overlay, direction_angle, detected_markers

# ── WebRTC & Loop (Condensed) ────────────────────────────────────────────────

def start_webrtc(frame_queue: Queue):
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="unitree.local")
    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            if not frame_queue.empty(): frame_queue.get_nowait()
            frame_queue.put(img)
    async def setup():
        await conn.connect()
        conn.video.switchVideoChannel(True)
        conn.video.add_track_callback(recv_camera_stream)
    loop = asyncio.new_event_loop()
    threading.Thread(target=lambda: (asyncio.set_event_loop(loop), loop.run_until_complete(setup()), loop.run_forever()), daemon=True).start()
    return loop

def main():
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    frame_queue = Queue(maxsize=1)
    loop = start_webrtc(frame_queue)

    rclpy.init()
    ros_node = rclpy.create_node("unitree_vision_provider")
    heading_pub = ros_node.create_publisher(String, ROS_TOPIC, 10)
    frame_id = 0
    next_save_at = time.time()

    try:
        while rclpy.ok():
            if frame_queue.empty(): continue

            img = frame_queue.get()
            overlay, direction_angle, markers = process_frame(img)
            heading_rad = float(np.deg2rad(90.0 - direction_angle))

            # JSON Data includes ArUco markers
            msg_data = {
                "heading": heading_rad,
                "frame_id": frame_id,
                "markers": markers
            }
            msg = String(data=json.dumps(msg_data))
            heading_pub.publish(msg)

            if frame_id % PRINT_EVERY_N_FRAMES == 0:
                marker_log = f" | Markers: {markers}" if markers else ""
                print(f"[PUB] frame={frame_id} | head={heading_rad:.3f}{marker_log}", flush=True)

            if SHOW_PREVIEW:
                cv2.imshow("Vision", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"): break

            if time.time() >= next_save_at:
                cv2.imwrite(str(RECORDS_DIR / f"frame_{frame_id}.jpg"), overlay)
                next_save_at = time.time() + SAVE_INTERVAL_SEC

            frame_id += 1
            rclpy.spin_once(ros_node, timeout_sec=0.0)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
