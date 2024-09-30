import unittest
from boat_detection.comparison.comparator import Comparator
from boat_detection.database.db_manager import DatabaseManager
from boat_detection.utils.helpers import ensure_directory


class TestComparator(unittest.TestCase):
    def setUp(self):
        self.db_manager = DatabaseManager(db_path='data/databases/test_boats.db')
        self.db_manager.initialize_database()
        self.comparator = Comparator(
            db_manager=self.db_manager,
            results_dir='data/results/detection_images'
        )

    def tearDown(self):
        # TODO: Clean up test database
        self.db_manager.close()
        # TODO: Optionally remove test database file

    def test_orb_similarity(self):
        # TODO: Add tests for ORB similarity
        pass

    def test_structural_similarity(self):
        # TODO: Add tests for SSIM
        pass

    def test_perform_comparisons(self):
        # TODO: Add tests for perform_comparisons method
        pass


if __name__ == '__main__':
    unittest.main()
