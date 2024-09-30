import unittest
from unittest.mock import patch, MagicMock
import os
# import cv2
import numpy as np
import logging

from boat_detection.tracking.video_tracker import VideoTracker


class TestVideoTracker(unittest.TestCase):
    def setUp(self):
        self.config = {
            'videos_dir': 'data/input/videos',
            'output_dir': 'data/output/processed_videos',
            'results_dir': 'data/results',
            'logs_dir': 'data/results/logs',
            'detection_images_dir': 'data/results/detection_images',
            'models_dir': 'data/models',
            'model_path': 'trainedPrototypewithCars2.pt',
            'database_path': 'data/databases/test_boats.db',
            'movement_threshold': 100,
            'valid_detection_count': 5
        }

        self.tracker = VideoTracker(self.config)

    def tearDown(self):
        self.tracker.close()

    @patch('boat_detection.tracking.video_tracker.YOLO')
    @patch('boat_detection.tracking.video_tracker.DatabaseManager')
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    def test_track_videos_single_boat_detection(self, mock_video_writer, mock_video_capture, mock_db_manager_class,
                                                mock_yolo_class):

        mock_db_manager = MagicMock()
        mock_db_manager_class.return_value = mock_db_manager

        mock_yolo = MagicMock()
        mock_yolo_class.return_value = mock_yolo

        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap

        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]

        mock_results = [
            MagicMock(),
            MagicMock(),
            MagicMock()
        ]

        mock_results[0].boxes.xywh.cpu.return_value = np.array(
            [[320, 240, 50, 50]])
        mock_results[0].boxes.id.int.return_value = np.array([1])

        mock_results[1].boxes.xywh.cpu.return_value = np.array([])
        mock_results[1].boxes.id = None

        mock_yolo.track.side_effect = [mock_results[0], mock_results[1]]

        mock_out = MagicMock()
        mock_video_writer.return_value = mock_out

        self.tracker.track_videos()

        mock_db_manager.insert_boat_record.assert_called_once_with(
            track_id=1,
            status='launched',
            launch_time=1 / self.config['valid_detection_count'],
            model='trainedPrototypewithCars2.pt',
            match_id=None
        )

        video_file = 'test_video.m4v'
        mock_video_capture.assert_called_with(os.path.join(self.config['videos_dir'], video_file))

        mock_video_writer.assert_called()

        mock_cap.release.assert_called()
        mock_out.release.assert_called()

    @patch('boat_detection.tracking.video_tracker.YOLO')
    @patch('boat_detection.tracking.video_tracker.DatabaseManager')
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    def test_track_videos_multiple_boat_detections(self, mock_video_writer, mock_video_capture, mock_db_manager_class,
                                                   mock_yolo_class):

        mock_db_manager = MagicMock()
        mock_db_manager_class.return_value = mock_db_manager

        mock_yolo = MagicMock()
        mock_yolo_class.return_value = mock_yolo

        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap

        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]

        mock_results = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock()
        ]

        for i in range(3):
            mock_results[i].boxes.xywh.cpu.return_value = np.array([[320, 240, 50, 50]])
            mock_results[i].boxes.id.int.return_value = np.array([1])

        for i in range(3, 5):
            mock_results[i].boxes.xywh.cpu.return_value = np.array([[100, 100, 40, 40]])
            mock_results[i].boxes.id.int.return_value = np.array([2])

        mock_yolo.track.side_effect = mock_results[:5]

        mock_out = MagicMock()
        mock_video_writer.return_value = mock_out

        self.tracker.track_videos()

        expected_calls = [
            unittest.mock.call(
                track_id=1,
                status='launched',
                launch_time=3 / self.config['valid_detection_count'],
                model='trainedPrototypewithCars2.pt',
                match_id=None
            ),
            unittest.mock.call(
                track_id=2,
                status='launched',
                launch_time=5 / self.config['valid_detection_count'],
                model='trainedPrototypewithCars2.pt',
                match_id=None
            )
        ]
        mock_db_manager.insert_boat_record.assert_has_calls(expected_calls, any_order=True)
        self.assertEqual(mock_db_manager.insert_boat_record.call_count, 2)

        mock_cap.release.assert_called()
        mock_out.release.assert_called()

    @patch('boat_detection.tracking.video_tracker.YOLO')
    @patch('boat_detection.tracking.video_tracker.DatabaseManager')
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    def test_track_videos_no_detections(self, mock_video_writer, mock_video_capture, mock_db_manager_class,
                                        mock_yolo_class):

        mock_db_manager = MagicMock()
        mock_db_manager_class.return_value = mock_db_manager

        mock_yolo = MagicMock()
        mock_yolo_class.return_value = mock_yolo

        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap

        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]

        mock_results = [
            MagicMock(),
            MagicMock()
        ]

        mock_results[0].boxes.xywh.cpu.return_value = np.array([])
        mock_results[0].boxes.id = None

        mock_results[1].boxes.xywh.cpu.return_value = np.array([])
        mock_results[1].boxes.id = None

        mock_yolo.track.side_effect = mock_results

        mock_out = MagicMock()
        mock_video_writer.return_value = mock_out

        self.tracker.track_videos()

        mock_db_manager.insert_boat_record.assert_not_called()

        mock_cap.release.assert_called()
        mock_out.release.assert_called()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
