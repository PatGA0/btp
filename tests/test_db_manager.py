import unittest
from boat_detection.database.db_manager import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.db_manager = DatabaseManager(db_path='data/databases/test_boats.db')
        self.db_manager.initialize_database()

    def tearDown(self):
        self.db_manager.close()
        # delete the test database file
        # FYI This is optional
        import os
        os.remove('data/databases/test_boats.db')

    def test_insert_multiple_records(self):
        self.db_manager.insert_boat_record(1, 'launched', 100.0, 'model_A')
        self.db_manager.insert_boat_record(2, 'launched', 200.0, 'model_B')
        records = self.db_manager.fetch_all_boat_records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][1], 1)
        self.assertEqual(records[1][1], 2)


if __name__ == '__main__':
    unittest.main()
