import unittest
from unittest.mock import patch, MagicMock
import os
# import shutil
# import cv2
import numpy as np

from boat_detection.utils.helpers import (
    calculate_movement,
    format_timestamp,
    ensure_directory,
    remove_directory,
    remove_files,
    load_environment,
    get_env_variable,
    setup_logging,
    load_config,
    save_detection_image,
    get_all_video_files,
    calculate_on_water_time
)


class TestHelpers(unittest.TestCase):
    def test_calculate_movement(self):
        start_pos = (0.0, 0.0)
        end_pos = (3.0, 4.0)
        distance = calculate_movement(start_pos, end_pos)
        self.assertEqual(distance, 5.0)

        start_pos = (10.5, 20.5)
        end_pos = (13.5, 24.5)
        distance = calculate_movement(start_pos, end_pos)
        self.assertAlmostEqual(distance, 5.0)

    def test_format_timestamp(self):
        timestamp = 3661.75  # 1 hour, 1 minute, 1.75 seconds
        formatted = format_timestamp(timestamp)
        self.assertEqual(formatted, "01:01:01.75")

        timestamp = 0.0
        formatted = format_timestamp(timestamp)
        self.assertEqual(formatted, "00:00:00.00")

        timestamp = 86399.99  # 23:59:59.99
        formatted = format_timestamp(timestamp)
        self.assertEqual(formatted, "23:59:59.99")

        timestamp = 12.3456
        formatted = format_timestamp(timestamp)
        self.assertEqual(formatted, "00:00:12.35")

    @patch('boat_detection.utils.helpers.os.makedirs')
    def test_ensure_directory_existing(self, mock_makedirs):
        path = 'test_dir'
        os.makedirs(path, exist_ok=True)
        ensure_directory(path)
        mock_makedirs.assert_called_with(path, exist_ok=True)

    @patch('boat_detection.utils.helpers.shutil.rmtree')
    @patch('boat_detection.utils.helpers.os.path.exists')
    def test_remove_directory_existing(self, mock_exists, mock_rmtree):
        mock_exists.return_value = True
        path = 'test_dir'
        remove_directory(path)
        mock_rmtree.assert_called_with(path)

    @patch('boat_detection.utils.helpers.shutil.rmtree')
    @patch('boat_detection.utils.helpers.os.path.exists')
    def test_remove_directory_nonexistent(self, mock_exists, mock_rmtree):
        mock_exists.return_value = False
        path = 'nonexistent_dir'
        with self.assertLogs(level='WARNING') as log:
            remove_directory(path)
            self.assertIn('Attempted to remove non-existent directory', log.output[0])
        mock_rmtree.assert_not_called()

    @patch('boat_detection.utils.helpers.shutil.rmtree')
    def test_remove_files(self, mock_rmtree):
        # Mock glob.glob to return a list of files
        with patch('boat_detection.utils.helpers.glob.glob', return_value=['file1.txt', 'file2.txt']):
            with patch('boat_detection.utils.helpers.os.remove') as mock_remove:
                remove_files('*.txt')
                mock_remove.assert_any_call('file1.txt')
                mock_remove.assert_any_call('file2.txt')
                self.assertEqual(mock_remove.call_count, 2)

    @patch('boat_detection.utils.helpers.cv2.imwrite')
    def test_save_detection_image_success(self, mock_imwrite):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        path = 'detection.jpg'
        save_detection_image(image, path)
        mock_imwrite.assert_called_with(path, image)

    @patch('boat_detection.utils.helpers.cv2.imwrite', side_effect=Exception('Save failed'))
    def test_save_detection_image_failure(self, mock_imwrite):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        path = 'detection.jpg'
        with self.assertRaises(Exception):
            save_detection_image(image, path)

    @patch('boat_detection.utils.helpers.os.listdir')
    def test_get_all_video_files(self, mock_listdir):
        mock_listdir.return_value = ['video1.mp4', 'video2.m4v', 'image.png']
        videos = get_all_video_files('videos_dir')
        self.assertEqual(videos, ['video1.mp4', 'video2.m4v'])

    @patch('boat_detection.utils.helpers.os.listdir')
    def test_get_all_video_files_no_matches(self, mock_listdir):
        mock_listdir.return_value = ['image1.png', 'document.pdf']
        videos = get_all_video_files('videos_dir')
        self.assertEqual(videos, [])

    def test_calculate_on_water_time_with_valid_times(self):
        with patch('boat_detection.utils.helpers.DatabaseManager') as mock_db_manager_class:
            mock_db_manager = MagicMock()
            mock_db_manager.get_boat_launch_time.return_value = 100.0
            mock_db_manager_class.return_value = mock_db_manager

            on_water_time = calculate_on_water_time(1, 200.0)
            self.assertEqual(on_water_time, 100.0)

    def test_calculate_on_water_time_missing_launch_time(self):
        with patch('boat_detection.utils.helpers.DatabaseManager') as mock_db_manager_class:
            mock_db_manager = MagicMock()
            mock_db_manager.get_boat_launch_time.return_value = None
            mock_db_manager_class.return_value = mock_db_manager

            on_water_time = calculate_on_water_time(1, 200.0)
            self.assertEqual(on_water_time, 0.0)

    def test_get_env_variable_with_default(self):
        with patch('boat_detection.utils.helpers.os.getenv', return_value=None):
            value = get_env_variable('NON_EXISTENT_VAR', default='default_value')
            self.assertEqual(value, 'default_value')

    def test_get_env_variable_cast_list(self):
        with patch('boat_detection.utils.helpers.os.getenv', return_value='item1,item2,item3'):
            value = get_env_variable('LIST_VAR', cast_type=list)
            self.assertEqual(value, ['item1', 'item2', 'item3'])

    def test_get_env_variable_cast_int_success(self):
        with patch('boat_detection.utils.helpers.os.getenv', return_value='42'):
            value = get_env_variable('INT_VAR', cast_type=int)
            self.assertEqual(value, 42)

    def test_get_env_variable_cast_int_failure(self):
        with patch('boat_detection.utils.helpers.os.getenv', return_value='not_an_int'):
            value = get_env_variable('INT_VAR', cast_type=int, default=0)
            self.assertEqual(value, 0)

    @patch('boat_detection.utils.helpers.yaml.safe_load')
    def test_load_config_success(self, mock_safe_load):
        mock_safe_load.return_value = {'key': 'value'}
        config = load_config('config.yaml')
        self.assertEqual(config, {'key': 'value'})
        mock_safe_load.assert_called_once()

    @patch('boat_detection.utils.helpers.open', side_effect=FileNotFoundError)
    def test_load_config_file_not_found(self, mock_open):
        with self.assertRaises(FileNotFoundError):
            load_config('nonexistent_config.yaml')

    @patch('boat_detection.utils.helpers.open', new_callable=unittest.mock.mock_open, read_data="invalid: [yaml")
    def test_load_config_yaml_error(self, mock_open):
        with self.assertRaises(Exception):
            load_config('invalid_config.yaml')

    @patch('boat_detection.utils.helpers.os.makedirs', side_effect=Exception('Creation failed'))
    def test_ensure_directory_failure(self, mock_makedirs):
        with self.assertRaises(Exception):
            ensure_directory('invalid_path')

    @patch('boat_detection.utils.helpers.logging.basicConfig')
    def test_setup_logging(self, mock_basicConfig):
        setup_logging('logfile.log')
        mock_basicConfig.assert_called_with(
            filename='logfile.log',
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )


if __name__ == '__main__':
    unittest.main()
