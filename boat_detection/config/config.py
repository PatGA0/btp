import os
import yaml
from dotenv import load_dotenv
from typing import List, Optional
import logging

class Config:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = {}
        self.load_environment()
        self.load_config()

    def load_environment(self):
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', '.env')
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"Environment variables loaded from {dotenv_path}.")

    def load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                logging.info(f"Configuration loaded from {self.config_path}.")
        except FileNotFoundError:
            logging.error(f"Configuration file {self.config_path} not found.")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing the configuration file: {e}")
            raise

    def get(self, key: str, default: Optional[object] = None) -> Optional[object]:
        return self.config.get(key, default)

    # Example property methods for easy access
    @property
    def videos_dir(self) -> str:
        return self.get('videos_dir', 'data/input/videos')

    @property
    def output_dir(self) -> str:
        return self.get('output_dir', 'data/output/processed_videos')

    @property
    def results_dir(self) -> str:
        return self.get('results_dir', 'data/results')

    @property
    def logs_dir(self) -> str:
        return self.get('logs_dir', 'data/results/logs')

    @property
    def detection_images_dir(self) -> str:
        return self.get('detection_images_dir', 'data/results/detection_images')

    @property
    def models_dir(self) -> str:
        return self.get('models_dir', 'data/models')

    @property
    def model_path(self) -> str:
        return self.get('model_path', 'trainedPrototypewithCars2.pt')

    @property
    def database_dir(self) -> str:
        return self.get('database_dir', 'data/databases')

    @property
    def database_path(self) -> str:
        return self.get('database_path', os.path.join(self.database_dir, 'boats.db'))

    @property
    def movement_threshold(self) -> int:
        return self.get('movement_threshold', 100)

    @property
    def valid_detection_count(self) -> int:
        return self.get('valid_detection_count', 5)

    @property
    def orb_threshold(self) -> float:
        return self.get('orb_threshold', 0.3)

    @property
    def ssim_threshold(self) -> float:
        return self.get('ssim_threshold', 0.1)

    @property
    def time_threshold(self) -> float:
        return self.get('time_threshold', 1800)

    @property
    def nc(self) -> int:
        return self.get('nc', 0)

    @property
    def names(self) -> List[str]:
        return self.get('names', [])

# from boat_detection.config.config import Config
# config = Config()
# print(config.videos_dir)
