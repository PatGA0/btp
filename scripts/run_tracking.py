# import sys
import os
import logging
from boat_detection.tracking.video_tracker import VideoTracker
from boat_detection.utils.helpers import load_config, setup_logging


def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    config = load_config(config_path)

    tracker = VideoTracker(config)

    try:
        tracker.run()
    except Exception as e:
        logging.error(f"An error occurred during video tracking: {e}")
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
