import os
import cv2
# import sqlite3
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import logging
import time
import shutil
import math

from boat_detection.utils.helpers import setup_logging, ensure_directory, get_env_variable, load_environment
from boat_detection.database.db_manager import DatabaseManager
# from boat_detection.comparison.comparator import Comparator


class VideoTracker:
    def __init__(self, config: dict):
        self.config = config
        load_environment()

        self.videos_dir = config['videos_dir']
        self.output_dir = config['output_dir']
        self.results_dir = config['results_dir']
        self.logs_dir = config['logs_dir']
        self.detection_images_dir = config['detection_images_dir']
        self.models_dir = config['models_dir']
        self.model_path = config['model_path']

        self.movement_threshold = config.get('movement_threshold', 100)
        self.valid_detection_count = config.get('valid_detection_count', 5)

        log_file = os.path.join(self.logs_dir, 'processing.log')
        setup_logging(log_file)

        ensure_directory(self.output_dir)
        ensure_directory(self.detection_images_dir)
        ensure_directory(self.results_dir)
        ensure_directory(self.logs_dir)

        self.db_manager = DatabaseManager(db_path=config['database_path'])
        self.db_manager.initialize_database()

        self.model = self.load_model()

    def load_model(self):
        model_full_path = os.path.join(self.models_dir, self.model_path)
        try:
            model = YOLO(model_full_path)
            logging.info(f"YOLO model loaded from {model_full_path}.")
            return model
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise

    def save_boat_to_db(self, track_id: int, status: str, timestamp: float, model_name: str, match_id: int = None):
        self.db_manager.insert_boat_record(track_id, status, timestamp, model_name, match_id)

    def update_boat_in_db(self, track_id: int, status: str, timestamp: float, match_id: int = None):
        self.db_manager.update_boat_record(track_id, status, timestamp,
                                           self.calculate_on_water_time(track_id, timestamp), match_id)

    def calculate_on_water_time(self, track_id: int, retrieve_time: float) -> float:
        launch_time = self.db_manager.get_boat_launch_time(track_id)
        if launch_time is not None:
            return retrieve_time - launch_time
        else:
            logging.warning(f"Cannot calculate on-water time for Track ID={track_id} as launch time is missing.")
            return 0.0

    def remove_detection_images(self, track_id: int):
        track_folder = os.path.join(self.detection_images_dir, f"track_id_{track_id}")
        if os.path.exists(track_folder):
            try:
                shutil.rmtree(track_folder)
                logging.info(f"Removed detection images for track ID {track_id}.")
            except Exception as e:
                logging.error(f"Failed to remove detection images for track ID {track_id}: {e}")
        else:
            logging.warning(f"No detection images found for track ID {track_id} to remove.")

    def track_videos(self):
        video_files = [f for f in os.listdir(self.videos_dir) if f.endswith(('.m4v', '.mp4'))]
        if not video_files:
            logging.error("No video files found in the videos directory.")
            return

        for video_file in video_files:
            video_path = os.path.join(self.videos_dir, video_file)
            output_video_path = os.path.join(self.output_dir, f"output_{os.path.splitext(video_file)[0]}.mp4")
            logging.info(f"Processing video: {video_file}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Cannot open video file: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            new_width = math.ceil(frame_width / 32) * 32
            new_height = math.ceil(frame_height / 32) * 32

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

            track_history = defaultdict(lambda: {'detections': 0, 'positions': []})
            valid_tracks = set()
            boat_records = {}

            frame_number = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                current_time_sec = frame_number / fps

                frame_resized = cv2.resize(frame, (new_width, new_height))

                try:
                    results = self.model.track(frame_resized, persist=True, imgsz=(new_width, new_height), conf=0.5)
                except Exception as e:
                    logging.error(f"YOLO tracking failed at frame {frame_number} in {video_file}: {e}")
                    continue

                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes.xywh.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().numpy() if result.boxes.id is not None else []

                    for box, track_id in zip(boxes, track_ids):
                        x_center, y_center, w, h = box
                        track_id = int(track_id)

                        history = track_history[track_id]
                        history['detections'] += 1
                        history['positions'].append((x_center, y_center))
                        if len(history['positions']) > self.valid_detection_count:
                            history['positions'].pop(0)

                        if history['detections'] >= self.valid_detection_count:
                            start_pos = np.array(history['positions'][0])
                            current_pos = np.array(history['positions'][-1])
                            movement = np.linalg.norm(current_pos - start_pos)

                            if movement >= self.movement_threshold:
                                valid_tracks.add(track_id)

                                if track_id not in boat_records:
                                    boat_records[track_id] = 'launched'
                                    self.save_boat_to_db(track_id, 'launched', current_time_sec,
                                                         os.path.basename(self.model_path))
                                elif boat_records[track_id] == 'launched':
                                    boat_records[track_id] = 'retrieved'
                                    self.update_boat_in_db(track_id, 'retrieved', current_time_sec)
                                    self.remove_detection_images(track_id)

                        x1 = int(x_center - w / 2)
                        y1 = int(y_center - h / 2)
                        x2 = int(x_center + w / 2)
                        y2 = int(y_center + h / 2)
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_resized, f'ID: {track_id}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if len(history['positions']) >= 2:
                            for idx in range(1, len(history['positions'])):
                                pt1 = (int(history['positions'][idx - 1][0]), int(history['positions'][idx - 1][1]))
                                pt2 = (int(history['positions'][idx][0]), int(history['positions'][idx][1]))
                                cv2.line(frame_resized, pt1, pt2, (255, 0, 0), 2)

                        if track_id not in boat_records:
                            track_folder = os.path.join(self.detection_images_dir, f"track_id_{track_id}")
                            ensure_directory(track_folder)
                            frame_filename = f"frame_{frame_number:04d}.jpg"
                            frame_path = os.path.join(track_folder, frame_filename)
                            cv2.imwrite(frame_path, frame_resized)
                            logging.info(f"Saved detection image: {frame_path}")

                out.write(frame_resized)

                if frame_number % 100 == 0:
                    logging.info(f"Processed frame {frame_number}/{total_frames} in {video_file}.")

            cap.release()
            out.release()
            logging.info(f"Finished processing video: {video_file}. Output saved to {output_video_path}.")

    def close(self):
        self.db_manager.close()

    def run(self):
        start_time = time.time()
        logging.info("Boat tracking started.")
        self.track_videos()
        elapsed_time = time.time() - start_time
        logging.info(f"Boat tracking completed in {elapsed_time:.2f} seconds.")
