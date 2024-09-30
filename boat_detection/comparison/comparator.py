import os
import fnmatch
import cv2
import shutil
import logging
from skimage.metrics import structural_similarity
from boat_detection.database.db_manager import DatabaseManager
from boat_detection.utils.helpers import ensure_directory

class Comparator:
    def __init__(self, db_manager: DatabaseManager, results_dir: str,
                 orb_threshold: float = 0.3, ssim_threshold: float = 0.1, time_threshold: float = 1800):
        self.db_manager = db_manager
        self.results_dir = results_dir
        self.orb_threshold = orb_threshold
        self.ssim_threshold = ssim_threshold
        self.time_threshold = time_threshold

        self.match_dir = os.path.join(self.results_dir, 'matches')
        self.dupe_dir = os.path.join(self.results_dir, 'duplicates')
        self.orphans_dir = os.path.join(self.results_dir, 'orphans')
        ensure_directory(self.match_dir)
        ensure_directory(self.dupe_dir)
        ensure_directory(self.orphans_dir)

    @staticmethod
    def orb_sim(img1, img2) -> float:
        orb = cv2.ORB_create()
        kp_a, desc_a = orb.detectAndCompute(img1, None)
        kp_b, desc_b = orb.detectAndCompute(img2, None)

        if desc_a is None or desc_b is None:
            logging.debug("One of the images has no descriptors. ORB similarity set to 0.")
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc_a, desc_b)
        similar_regions = [m for m in matches if m.distance < 50]

        if len(matches) == 0:
            return 0.0

        similarity = len(similar_regions) / len(matches)
        logging.debug(f"ORB similarity: {similarity:.4f}")
        return similarity

    @staticmethod
    def structural_sim(img1, img2) -> float:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if img1_gray.shape != img2_gray.shape:
            img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))

        sim, _ = structural_similarity(img1_gray, img2_gray, full=True)
        logging.debug(f"SSIM similarity: {sim:.4f}")
        return sim

    def perform_comparisons(self):
        logging.info("Starting perform_comparisons.")

        track_folders = [f for f in os.listdir(self.results_dir) if
                         f.startswith('track_id_') and os.path.isdir(os.path.join(self.results_dir, f))]
        track_ids = [int(f.split('_')[2]) for f in track_folders if f.split('_')[2].isdigit()]
        logging.info(f"Found {len(track_ids)} track_ids for comparison.")

        processed = set()

        for i, track_id in enumerate(track_ids):
            if track_id in processed:
                continue

            track_id_dir = os.path.join(self.results_dir, f"track_id_{track_id}")
            images_i = fnmatch.filter(os.listdir(track_id_dir), '*.jpg')
            images_i_paths = [os.path.join(track_id_dir, img) for img in images_i]

            if not images_i_paths:
                logging.warning(f"No images found for track_id {track_id}. Skipping.")
                continue

            img1 = cv2.imread(images_i_paths[0])
            if img1 is None:
                logging.warning(f"Failed to read image {images_i_paths[0]}. Skipping track_id {track_id}.")
                continue

            for j in range(i + 1, len(track_ids)):
                compare_id = track_ids[j]

                if compare_id in processed:
                    continue

                compare_id_dir = os.path.join(self.results_dir, f"track_id_{compare_id}")
                if not os.path.isdir(compare_id_dir):
                    logging.warning(f"Track ID directory {compare_id_dir} does not exist. Skipping.")
                    continue

                images_j = fnmatch.filter(os.listdir(compare_id_dir), '*.jpg')
                images_j_paths = [os.path.join(compare_id_dir, img) for img in images_j]

                if not images_j_paths:
                    logging.warning(f"No images found for compare_id {compare_id}. Skipping.")
                    continue

                img2 = cv2.imread(images_j_paths[0])
                if img2 is None:
                    logging.warning(f"Failed to read image {images_j_paths[0]}. Skipping compare_id {compare_id}.")
                    continue

                orb_score = self.orb_sim(img1, img2)
                ssim_score = self.structural_sim(img1, img2)

                if orb_score > self.orb_threshold and ssim_score > self.ssim_threshold:
                    launch_time_i = self.db_manager.get_boat_launch_time(track_id)
                    launch_time_j = self.db_manager.get_boat_launch_time(compare_id)

                    if launch_time_i is None or launch_time_j is None:
                        logging.warning(f"Missing launch times for Track IDs {track_id} or {compare_id}. Skipping.")
                        continue

                    time_diff = abs(launch_time_i - launch_time_j)

                    if time_diff > self.time_threshold:
                        logging.info(f"Boat ID {track_id} matched with Boat ID {compare_id} (Time diff: {time_diff}s).")

                        self.db_manager.cursor.execute('''
                            UPDATE boats
                            SET matchID = ?, status = 'Match'
                            WHERE track_id = ?
                        ''', (compare_id, track_id))
                        self.db_manager.conn.commit()

                        for img in images_i:
                            src = os.path.join(track_id_dir, img)
                            dest = os.path.join(self.match_dir, f"track_id_{track_id}_{img}")
                            shutil.move(src, dest)

                        for img in images_j:
                            src = os.path.join(compare_id_dir, img)
                            dest = os.path.join(self.match_dir, f"track_id_{compare_id}_{img}")
                            shutil.move(src, dest)

                        processed.add(compare_id)
                        break

                    else:
                        logging.info(
                            f"Boat ID {track_id} marked as Duplicate of Boat ID {compare_id} (Time diff: {time_diff}s).")

                        self.db_manager.cursor.execute('''
                            UPDATE boats
                            SET status = 'Duplicate', matchID = ?
                            WHERE track_id = ?
                        ''', (compare_id, track_id))
                        self.db_manager.conn.commit()

                        for img in images_i:
                            src = os.path.join(track_id_dir, img)
                            dest = os.path.join(self.dupe_dir, f"track_id_{track_id}_{img}")
                            shutil.move(src, dest)

                        for img in images_j:
                            src = os.path.join(compare_id_dir, img)
                            dest = os.path.join(self.dupe_dir, f"track_id_{compare_id}_{img}")
                            shutil.move(src, dest)

                        processed.add(track_id)
                        break

        for track_id in track_ids:
            if track_id not in processed:
                self.db_manager.cursor.execute('SELECT status FROM boats WHERE track_id = ?', (track_id,))
                result = self.db_manager.cursor.fetchone()
                if result and result[0] not in ['Retrieved', 'Match']:
                    self.db_manager.cursor.execute('''
                        UPDATE boats
                        SET status = 'Orphan'
                        WHERE track_id = ?
                    ''', (track_id,))
                    self.db_manager.conn.commit()
                    logging.info(f"Boat ID {track_id} marked as Orphan.")

                    track_id_dir = os.path.join(self.results_dir, f"track_id_{track_id}")
                    images = fnmatch.filter(os.listdir(track_id_dir), '*.jpg')
                    for img in images:
                        src = os.path.join(track_id_dir, img)
                        dest = os.path.join(self.orphans_dir, f"track_id_{track_id}_{img}")
                        shutil.move(src, dest)

        logging.info("Completed perform_comparisons.")
