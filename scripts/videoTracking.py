import os
import cv2
import sqlite3
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import logging
import time
import shutil

from utilities import (
    setup_logging,
    ensure_directory,
    get_env_variable,
    load_environment
)
from comparison import perform_comparisons, orb_sim, structural_sim
# from database_setup import get_boat_launch_time

# Just loading the Env variables from the .env.
load_environment()

BASE_DIR = get_env_variable('BASE_DIR')
VIDEOS_DIR = get_env_variable('VIDEOS_DIR')
OUTPUT_DIR = get_env_variable('OUTPUT_DIR')
RESULTS_DIR = get_env_variable('RESULTS_DIR')
LOGS_DIR = get_env_variable('LOGS_DIR')
DETECTION_IMAGES_DIR = get_env_variable('DETECTION_IMAGES_DIR')
DATABASE_DIR = get_env_variable('DATABASE_DIR')
DATABASE_PATH = get_env_variable('DATABASE_PATH')
MODELS_DIR = get_env_variable('MODELS_DIR')
MODEL_PATH = get_env_variable('MODEL_PATH')

MOVEMENT_THRESHOLD = get_env_variable('MOVEMENT_THRESHOLD', cast_type=int)
VALID_DETECTION_COUNT = get_env_variable('VALID_DETECTION_COUNT', cast_type=int)

NC = get_env_variable('NC', cast_type=int)
NAMES = get_env_variable('NAMES', cast_type=list)

LOG_FILE = os.path.join(LOGS_DIR, 'processing.log')
setup_logging(LOG_FILE)


def initialize_database(db_path):
    logging.info(f"Connecting to database at: {db_path}")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS boats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER UNIQUE,
                status TEXT,
                launch_time REAL,
                retrieve_time REAL,
                on_water_time REAL,
                matchID INTEGER,
                model TEXT
            )
        ''')
        conn.commit()
        logging.info("Database initialized and boats table ensured.")
        return conn, cursor
    except sqlite3.Error as e:
        logging.error(f"Database initialization failed: {e}")
        raise


def save_boat_to_db(conn, cursor, track_id, status, timestamp, model_name, match_id=None):
    try:
        cursor.execute('''
            INSERT INTO boats (track_id, status, launch_time, model, matchID)
            VALUES (?, ?, ?, ?, ?)
        ''', (track_id, status, timestamp, model_name, match_id))
        conn.commit()
        logging.info(f"Boat ID {track_id} saved to database with status '{status}' at {timestamp}.")
    except sqlite3.IntegrityError:
        logging.warning(f"Boat ID {track_id} already exists in the database.")
    except sqlite3.Error as e:
        logging.error(f"Failed to insert boat ID {track_id}: {e}")
        raise


def update_boat_in_db(conn, cursor, track_id, status, timestamp, match_id=None):
    try:
        cursor.execute('SELECT launch_time FROM boats WHERE track_id = ?', (track_id,))
        result = cursor.fetchone()
        if result:
            launch_time = result[0]
            on_water_time = timestamp - launch_time
            cursor.execute('''
                UPDATE boats
                SET status = ?, retrieve_time = ?, on_water_time = ?, matchID = ?
                WHERE track_id = ?
            ''', (status, timestamp, on_water_time, match_id, track_id))
            conn.commit()
            logging.info(
                f"Boat ID {track_id} updated to status '{status}' at {timestamp} with on-water time {on_water_time} seconds.")
        else:
            logging.error(f"Boat ID {track_id} not found in database for update.")
    except sqlite3.Error as e:
        logging.error(f"Failed to update boat ID {track_id}: {e}")


def close_database(conn):
    try:
        conn.close()
        logging.info("Database connection closed.")
    except sqlite3.Error as e:
        logging.error(f"Error closing database: {e}")


def remove_detection_images(track_id):
    track_folder = os.path.join(DETECTION_IMAGES_DIR, f"track_id_{track_id}")
    if os.path.exists(track_folder):
        try:
            shutil.rmtree(track_folder)
            logging.info(f"Removed detection images for track ID {track_id}.")
        except Exception as e:
            logging.error(f"Failed to remove detection images for track ID {track_id}: {e}")
    else:
        logging.warning(f"No detection images found for track ID {track_id} to remove.")


def track_videos():
    conn, cursor = initialize_database(DATABASE_PATH)

    model_path = os.path.join(MODELS_DIR, 'trainedPrototypewithCars2.pt')
    try:
        model = YOLO(model_path)
        logging.info(f"YOLO model loaded from {model_path}.")
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        close_database(conn)
        return

    video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith('.m4v')]
    if not video_files:
        logging.error("No .m4v video files found in the videos directory.")
        close_database(conn)
        return

    for video_file in video_files:
        video_path = os.path.join(VIDEOS_DIR, video_file)
        output_video_path = os.path.join(OUTPUT_DIR, f"output_{video_file.replace('.m4v', '.mp4')}")
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
                results = model.track(frame_resized, persist=True, imgsz=(new_width, new_height), conf=0.5)
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
                    if len(history['positions']) > VALID_DETECTION_COUNT:
                        history['positions'].pop(0)

                    if history['detections'] >= VALID_DETECTION_COUNT:
                        start_pos = np.array(history['positions'][0])
                        current_pos = np.array(history['positions'][-1])
                        movement = np.linalg.norm(current_pos - start_pos)

                        if movement >= MOVEMENT_THRESHOLD:
                            valid_tracks.add(track_id)

                            if track_id not in boat_records:
                                boat_records[track_id] = 'launched'
                                save_boat_to_db(conn, cursor, track_id, 'launched', current_time_sec,
                                                os.path.basename(model_path))
                            elif boat_records[track_id] == 'launched':
                                # Boat is being retrieved
                                boat_records[track_id] = 'retrieved'
                                update_boat_in_db(conn, cursor, track_id, 'retrieved', current_time_sec)
                                remove_detection_images(track_id)

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
                        track_folder = os.path.join(DETECTION_IMAGES_DIR, f"track_id_{track_id}")
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

    perform_comparisons(conn, cursor, DETECTION_IMAGES_DIR)

    close_database(conn)
    logging.info("All videos processed and comparisons completed successfully.")


if __name__ == "__main__":
    import math  # Import math for image size adjustments

    start_time = time.time()
    logging.info("Boat tracking started.")
    track_videos()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Boat tracking completed in {elapsed_time:.2f} seconds.")
