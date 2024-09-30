# import sys
import os
import logging
from boat_detection.comparison.comparator import Comparator
from boat_detection.database.db_manager import DatabaseManager
from boat_detection.utils.helpers import load_config, setup_logging


def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    config = load_config(config_path)

    db_manager = DatabaseManager(db_path=config['database_path'])

    comparator = Comparator(db_manager=db_manager, results_dir=config['detection_images_dir'])

    try:
        comparator.perform_comparisons()
    except Exception as e:
        logging.error(f"An error occurred during comparisons: {e}")
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
