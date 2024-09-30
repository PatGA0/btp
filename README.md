# Boat Tracking Project

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Initializing the Database](#initializing-the-database)
  - [Processing Videos](#processing-videos)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The **Boat Tracking Project** is an automated system designed to detect, track, and record the movements of boats in video footage. Leveraging the power of the YOLO (You Only Look Once) object detection model and a SQLite database, this project provides a comprehensive solution for monitoring boat activities in various maritime environments.

## Features

- **Real-Time Boat Detection:** Utilizes the YOLO model for efficient and accurate detection of boats in video frames.
- **Unique Tracking:** Assigns unique `track_id`s to each detected boat, enabling precise tracking over time.
- **Status Recording:** Automatically records the launch and retrieval times of boats, along with their on-water durations.
- **Database Management:** Stores all relevant boat data in a structured SQLite database for easy access and analysis.
- **Comprehensive Logging:** Maintains detailed logs of all operations for monitoring, debugging, and auditing purposes.
- **Modular Design:** Organized into distinct scripts and utilities for maintainability and scalability.

## Project Structure

```
BoatTrackingProject/
    ├── config.yaml
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── scripts/
    │   ├── videoTracking.py
    │   ├── database_setup.py
    │   └── utilities.py
    ├── models/
    │   └── trainedPrototypewithCars2.pt
    ├── data/
    │   ├── videos/
    │   │   ├── video1.m4v
    │   │   ├── video2.m4v
    │   │   └── ...
    │   ├── output/
    │   │   ├── output_video1.mp4
    │   │   ├── output_video2.mp4
    │   │   └── ...
    │   └── results/
    │       ├── detection_images/
    │       │   ├── track_id_1/
    │       │   ├── track_id_2/
    │       │   └── ...
    │       └── logs/
    │           ├── processing.log
    │           ├── database_setup.log
    │           └── utilities.log
    └── database/
        └── boats.db
```

- **config.yaml:** Configuration file containing project parameters.
- **requirements.txt:** Lists all Python dependencies required for the project.
- **scripts/:** Contains all Python scripts:
  - `videoTracking.py`: Main script for processing videos and tracking boats.
  - `database_setup.py`: Initializes the SQLite database and sets up tables.
  - `utilities.py`: Helper functions for various tasks.
- **models/:** Stores the trained YOLO model (`.pt` file).
- **data/:**
  - `videos/`: Input video files to be processed.
  - `output/`: Annotated output videos with detections and tracking.
  - `results/`: Contains detection images and log files.
- **database/:**
  - `boats.db`: SQLite database storing boat records.

## Prerequisites

Before setting up the Boat Tracking Project, ensure you have the following installed:

- **Python 3.7 or higher**: [Download Python](https://www.python.org/downloads/)
- **Git** (optional, for version control): [Download Git](https://git-scm.com/downloads)
- **Virtual Environment Tool** (recommended): Python's built-in `venv` or `virtualenv`

## Installation

### 1. Clone the Repository

If you haven't already, clone the repository to your local machine:

```bash
git clone https://github.com/Slipstream84/BoatTrackingProject.git
cd BoatTrackingProject
```

*Replace `https://github.com/yourusername/BoatTrackingProject.git` with the actual repository URL.*

### 2. Create a Virtual Environment (Recommended)

Creating a virtual environment helps manage project-specific dependencies without affecting your global Python installation.

```bash
# On Windows
python -m venv venv

# On macOS/Linux
python3 -m venv venv
```

### 3. Activate the Virtual Environment

```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

*After activation, your terminal prompt should change to indicate that the virtual environment is active.*

### 4. Install Required Packages

Ensure that `requirements.txt` contains all necessary dependencies. A sample `requirements.txt` based on the provided scripts might look like this:

*Note:* Adjust versions as necessary based on compatibility.

Install the dependencies using pip:
```bash
pip install -r requirements.txt
```

## Configuration

### `config.yaml`

The `config.yaml` file holds essential configuration parameters for the project. Below is a sample configuration:

```yaml
# config.yaml

path: D:\m4Qbt\BoatTrackingProject
train: train
val: val

nc: 5
names: ['Fishing Boat', 'Runabout', 'Sailboat', 'Yacht', 'Skiff']

# Additional configurations
movement_threshold: 20  # Threshold in pixels to consider movement
valid_detection_count: 5  # Number of detections required to validate a boat
```

**Parameters Explained:**

- **path:** Base directory path of the project.
- **train & val:** Directories related to model training and validation.
- **nc:** Number of classes the model can detect.
- **names:** Names of the classes (boat types) the model can identify.
- **movement_threshold:** Minimum pixel movement to consider a boat as moving.
- **valid_detection_count:** Number of consecutive detections required to validate a boat's presence.

*Ensure that the paths and parameters align with your project's directory structure and requirements.*

## Usage

### Initializing the Database

Before processing any videos, set up the SQLite database to store boat information.

1. **Run `database_setup.py`:**

   ```bash
   python scripts/database_setup.py
   ```

2. **Expected Outcome:**

   - Creates the `boats.db` SQLite database in the `database/` directory if it doesn't already exist.
   - Initializes the `boats` table with the required schema.
   - Logs the operation in `data/results/logs/database_setup.log`.

### Processing Videos

With the database initialized, you can now process your videos to detect and track boats.

1. **Prepare Your Videos:**

   Place all the `.m4v` (or supported format) video files you want to process in the `data/videos/` directory.

   **Example:**

   ```
   BoatTrackingProject/
   └── data/
       └── videos/
           ├── harbor1.m4v
           ├── harbor2.m4v
           └── ...
   ```

2. **Run `videoTracking.py`:**

   ```bash
   python scripts/videoTracking.py
   ```

3. **Processing Details:**

   - **Detection:** The script reads each video frame-by-frame and uses the YOLO model to detect boats.
   - **Tracking:** Assigns unique `track_id`s to each detected boat and tracks their movement across frames.
   - **Validation:** Validates boat presence based on movement thresholds and detection counts.
   - **Database Interaction:** Inserts new boat records upon detection and updates existing records upon retrieval.
   - **Output:** Annotated videos with detections and tracking lines are saved in `data/output/`.
   - **Logging:** Detailed logs are maintained in `data/results/logs/processing.log`.

4. **Post-Processing:**

   - **Annotated Videos:** Check the `data/output/` directory for the processed videos with detections.
   - **Detection Images:** Individual detection frames are stored in `data/results/detection_images/track_id_X/`.
   - **Database Records:** Verify boat records in `database/boats.db` using SQLite tools.

## Logging

Comprehensive logging is implemented to monitor the system's operations and facilitate debugging.

- **Log Files Location:**

  ```
  BoatTrackingProject/
  └── data/
      └── results/
          └── logs/
              ├── processing.log
              ├── database_setup.log
              └── utilities.log
  ```

- **Log Levels:**
  - **INFO:** General information about the system's operations.
  - **WARNING:** Potential issues that do not halt the system.
  - **ERROR:** Critical issues that may prevent certain operations from completing.

- **Example Log Entries:**

  ```
  2024-09-28 01:28:31,063 - INFO - Boat tracking started.
  2024-09-28 01:28:36,387 - INFO - Boat ID 2 saved to database with status 'launched' at 0.28.
  2024-09-28 01:28:36,836 - INFO - Boat ID 2 updated to status 'retrieved' at 0.32 with on-water time 0.04 seconds.
  ```