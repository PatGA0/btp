import glob
import cv2
import shutil
import yaml
import logging
import math
import os
from typing import Tuple, List, Optional
from dotenv import load_dotenv

CONFIG_FILE = 'config.yaml'


def load_environment():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, '..', '.env')
    load_dotenv(dotenv_path=env_path)


def get_env_variable(var_name, default=None, cast_type=str, separator=','):
    value = os.getenv(var_name, default)
    if value is None:
        return None
    if cast_type == list:
        return [item.strip() for item in value.split(separator)]
    try:
        return cast_type(value)
    except ValueError:
        return default


def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.info(f"Configuration loaded from {config_path}.")
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing the configuration file: {e}")
        raise


def ensure_directory(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Ensured directory exists: {path}.")
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        raise


def remove_directory(path: str) -> None:
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            logging.info(f"Removed directory and its contents: {path}.")
        else:
            logging.warning(f"Attempted to remove non-existent directory: {path}.")
    except Exception as e:
        logging.error(f"Failed to remove directory {path}: {e}")
        raise


def remove_files(pattern: str) -> None:
    try:
        files = glob.glob(pattern)
        for file in files:
            os.remove(file)
            logging.info(f"Removed file: {file}.")
    except Exception as e:
        logging.error(f"Failed to remove files with pattern {pattern}: {e}")
        raise


def calculate_movement(start_pos: Tuple[float, float], end_pos: Tuple[float, float]) -> float:
    distance = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
    logging.debug(f"Calculated movement from {start_pos} to {end_pos}: {distance} pixels.")
    return distance


def format_timestamp(seconds: float) -> str:
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        formatted_time = f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
        logging.debug(f"Formatted timestamp {seconds} seconds to {formatted_time}.")
        return formatted_time
    except Exception as e:
        logging.error(f"Failed to format timestamp {seconds}: {e}")
        raise


def save_detection_image(image: any, path: str) -> None:
    try:
        cv2.imwrite(path, image)
        logging.info(f"Saved detection image to {path}.")
    except Exception as e:
        logging.error(f"Failed to save image to {path}: {e}")
        raise


def get_all_video_files(videos_dir: str, extensions: Tuple[str, ...] = ('.m4v', '.mp4')) -> List[str]:
    try:
        video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith(extensions)]
        logging.info(f"Found {len(video_files)} video files in {videos_dir}.")
        return video_files
    except Exception as e:
        logging.error(f"Failed to list video files in {videos_dir}: {e}")
        raise


def calculate_on_water_time(launch_time: float, retrieve_time: float) -> float:
    on_water_time = retrieve_time - launch_time
    logging.debug(f"Calculated on-water time: {on_water_time} seconds.")
    return on_water_time
