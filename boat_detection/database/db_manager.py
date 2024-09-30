import os
import sqlite3
import logging
# from contextlib import closing
from typing import Optional, List

from boat_detection.utils.helpers import setup_logging, load_environment, get_env_variable

load_environment()

DATABASE_DIR = get_env_variable('DATABASE_DIR')
DATABASE_PATH = get_env_variable('DATABASE_PATH')
LOGS_DIR = get_env_variable('LOGS_DIR')
LOG_FILE = os.path.join(LOGS_DIR, 'database_setup.log')
setup_logging(LOG_FILE)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class DatabaseManager:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logging.info(f"Connected to database at {self.db_path}.")
        except sqlite3.Error as e:
            logging.error(f"Failed to connect to database: {e}")
            raise

    def initialize_database(self):
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS boats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER UNIQUE,
                    status TEXT NOT NULL,
                    launch_time REAL,
                    retrieve_time REAL,
                    on_water_time REAL,
                    matchID INTEGER,
                    model TEXT
                )
            ''')
            self.conn.commit()
            logging.info("Database initialized and 'boats' table created or already exists.")
        except sqlite3.Error as e:
            logging.error(f"Failed to initialize database: {e}")
            raise

    def insert_boat_record(self, track_id: int, status: str, launch_time: float, model: str, match_id: int = None):
        try:
            self.cursor.execute('''
                INSERT INTO boats (track_id, status, launch_time, model, matchID)
                VALUES (?, ?, ?, ?, ?)
            ''', (track_id, status, launch_time, model, match_id))
            self.conn.commit()
            logging.info(
                f"Inserted boat record: Track ID={track_id}, Status='{status}', Launch Time={launch_time}, Model='{model}', Match ID={match_id}.")
        except sqlite3.IntegrityError:
            logging.warning(f"Attempted to insert duplicate Track ID={track_id}. Operation skipped.")
        except sqlite3.Error as e:
            logging.error(f"Failed to insert boat record for Track ID={track_id}: {e}")
            raise

    def update_boat_record(self, track_id: int, status: str, retrieve_time: float, on_water_time: float, match_id: int = None):
        try:
            self.cursor.execute('''
                UPDATE boats
                SET status = ?, retrieve_time = ?, on_water_time = ?, matchID = ?
                WHERE track_id = ?
            ''', (status, retrieve_time, on_water_time, match_id, track_id))
            if self.cursor.rowcount == 0:
                logging.warning(f"No boat record found with Track ID={track_id} to update.")
            else:
                self.conn.commit()
                logging.info(
                    f"Updated boat record: Track ID={track_id}, Status='{status}', Retrieve Time={retrieve_time}, On-Water Time={on_water_time} seconds, Match ID={match_id}.")
        except sqlite3.Error as e:
            logging.error(f"Failed to update boat record for Track ID={track_id}: {e}")
            raise

    def get_boat_launch_time(self, track_id: int) -> Optional[float]:
        try:
            self.cursor.execute('SELECT launch_time FROM boats WHERE track_id = ?', (track_id,))
            result = self.cursor.fetchone()
            if result:
                logging.info(f"Retrieved launch time for Track ID={track_id}: {result[0]} seconds.")
                return result[0]
            else:
                logging.warning(f"No launch time found for Track ID={track_id}.")
                return None
        except sqlite3.Error as e:
            logging.error(f"Failed to retrieve launch time for Track ID={track_id}: {e}")
            raise

    def delete_boat_record(self, track_id: int):
        try:
            self.cursor.execute('DELETE FROM boats WHERE track_id = ?', (track_id,))
            if self.cursor.rowcount == 0:
                logging.warning(f"No boat record found with Track ID={track_id} to delete.")
            else:
                self.conn.commit()
                logging.info(f"Deleted boat record with Track ID={track_id}.")
        except sqlite3.Error as e:
            logging.error(f"Failed to delete boat record for Track ID={track_id}: {e}")
            raise

    def fetch_all_boat_records(self) -> List[tuple]:
        try:
            self.cursor.execute('SELECT * FROM boats')
            records = self.cursor.fetchall()
            logging.info(f"Fetched {len(records)} boat records from the database.")
            return records
        except sqlite3.Error as e:
            logging.error(f"Failed to fetch boat records: {e}")
            raise

    def close(self):
        try:
            if self.conn:
                self.conn.close()
                logging.info("Database connection closed.")
        except sqlite3.Error as e:
            logging.error(f"Error closing database: {e}")
