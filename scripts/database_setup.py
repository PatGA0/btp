import os
import sqlite3
import logging
from contextlib import closing

from utilities import (
    setup_logging,
    load_environment,
    get_env_variable
)

load_environment()

DATABASE_DIR = get_env_variable('DATABASE_DIR')
DATABASE_PATH = get_env_variable('DATABASE_PATH')
LOGS_DIR = get_env_variable('LOGS_DIR')
LOG_FILE = os.path.join(LOGS_DIR, 'database_setup.log')
setup_logging(LOG_FILE)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, LOG_FILE),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def initialize_database(db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute('''
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
                conn.commit()
                logging.info("Database initialized and 'boats' table created or already exists.")
    except sqlite3.Error as e:
        logging.error(f"Failed to initialize database: {e}")
        raise


def insert_boat_record(db_path, track_id, status, launch_time, model, match_id=None):
    try:
        with sqlite3.connect(db_path) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute('''
                    INSERT INTO boats (track_id, status, launch_time, model, matchID)
                    VALUES (?, ?, ?, ?, ?)
                ''', (track_id, status, launch_time, model, match_id))
                conn.commit()
                logging.info(
                    f"Inserted boat record: Track ID={track_id}, Status='{status}', Launch Time={launch_time}, Model='{model}', Match ID={match_id}.")
    except sqlite3.IntegrityError:
        logging.warning(f"Attempted to insert duplicate Track ID={track_id}. Operation skipped.")
    except sqlite3.Error as e:
        logging.error(f"Failed to insert boat record for Track ID={track_id}: {e}")
        raise



def update_boat_record(db_path, track_id, status, retrieve_time, on_water_time):
    try:
        with sqlite3.connect(db_path) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute('''
                    UPDATE boats
                    SET status = ?, retrieve_time = ?, on_water_time = ?
                    WHERE track_id = ?
                ''', (status, retrieve_time, on_water_time, track_id))
                if cursor.rowcount == 0:
                    logging.warning(f"No boat record found with Track ID={track_id} to update.")
                else:
                    conn.commit()
                    logging.info(
                        f"Updated boat record: Track ID={track_id}, Status='{status}', Retrieve Time={retrieve_time}, On-Water Time={on_water_time} seconds.")
    except sqlite3.Error as e:
        logging.error(f"Failed to update boat record for Track ID={track_id}: {e}")
        raise


def get_boat_launch_time(conn, track_id):
    """
    Retrieves the launch_time for a given track_id from the database.

    Parameters:
        conn (sqlite3.Connection): SQLite database connection.
        track_id (int): The track_id of the boat.

    Returns:
        float or None: The launch_time in seconds if found, else None.
    """
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT launch_time FROM boats WHERE track_id = ?', (track_id,))
        result = cursor.fetchone()
        if result:
            logging.info(f"Retrieved launch time for Track ID={track_id}: {result[0]} seconds.")
            return result[0]
        else:
            logging.warning(f"No launch time found for Track ID={track_id}.")
            return None
    except sqlite3.Error as e:
        logging.error(f"Failed to retrieve launch time for Track ID={track_id}: {e}")
        raise

def delete_boat_record(db_path, track_id):
    try:
        with sqlite3.connect(db_path) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute('DELETE FROM boats WHERE track_id = ?', (track_id,))
                if cursor.rowcount == 0:
                    logging.warning(f"No boat record found with Track ID={track_id} to delete.")
                else:
                    conn.commit()
                    logging.info(f"Deleted boat record with Track ID={track_id}.")
    except sqlite3.Error as e:
        logging.error(f"Failed to delete boat record for Track ID={track_id}: {e}")
        raise


def fetch_all_boat_records(db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute('SELECT * FROM boats')
                records = cursor.fetchall()
                logging.info(f"Fetched {len(records)} boat records from the database.")
                return records
    except sqlite3.Error as e:
        logging.error(f"Failed to fetch boat records: {e}")
        raise

def main():
    logging.info("Starting database setup...")
    initialize_database(DATABASE_PATH)
    logging.info("Database setup completed successfully.")


if __name__ == "__main__":
    main()
