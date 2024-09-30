# TODO: README.md

Sample project folder structure:
```
.
├── LICENSE
├── README.md
├── boat_detection
│   ├── __pycache__
│   ├── comparison
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── comparator.cpython-310.pyc
│   │   │   └── comparator.cpython-311.pyc
│   │   └── comparator.py
│   ├── config
│   │   ├── __init__.py
│   │   └── config.py
│   ├── database
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── db_manager.cpython-310.pyc
│   │   │   └── db_manager.cpython-311.pyc
│   │   └── db_manager.py
│   ├── tracking
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   └── video_tracker.cpython-310.pyc
│   │   └── video_tracker.py
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-310.pyc
│       │   ├── __init__.cpython-311.pyc
│       │   ├── helpers.cpython-310.pyc
│       │   └── helpers.cpython-311.pyc
│       └── helpers.py
├── config.yaml
├── data
│   ├── databases
│   │   ├── boats.db
│   │   └── test_boats.db
│   ├── input
│   │   └── videos
│   │       └── 06-02-2022 10 C Exmouth Marina.m4v
│   ├── models
│   │   └── trainedPrototypewithCars2.pt
│   ├── output
│   │   └── processed_videos
│   │       └── output_06-02-2022 10 C Exmouth Marina.mp4
│   └── results
│       ├── detection_images
│       │   ├── duplicates
│       │   ├── matches
│       │   ├── orphans
│       │   ├── track_id_1
│       │   ├── track_id_10
│       │   ├── track_id_12
│       │   ├── track_id_13
│       │   ├── track_id_14
│       │   ├── track_id_16
│       │   ├── track_id_17
│       │   ├── track_id_21
│       │   ├── track_id_25
│       │   ├── track_id_28
│       │   ├── track_id_3
│       │   ├── track_id_30
│       │   ├── track_id_31
│       │   ├── track_id_36
│       │   ├── track_id_38
│       │   ├── track_id_39
│       │   ├── track_id_41
│       │   ├── track_id_43
│       │   ├── track_id_47
│       │   ├── track_id_48
│       │   ├── track_id_5
│       │   ├── track_id_50
│       │   ├── track_id_52
│       │   ├── track_id_6
│       │   ├── track_id_65
│       │   ├── track_id_66
│       │   ├── track_id_7
│       │   ├── track_id_71
│       │   ├── track_id_74
│       │   ├── track_id_8
│       │   └── track_id_80
│       └── logs
│           └── database_setup.log
├── requirements.txt
├── scripts
│   ├── __pycache__
│   │   ├── perform_comparisons.cpython-310.pyc
│   │   ├── perform_comparisons.cpython-311.pyc
│   │   └── run_tracking.cpython-310.pyc
│   ├── perform_comparisons.py
│   └── run_tracking.py
├── setup.py
└── tests
    ├── __init__.py
    ├── __pycache__
    │   ├── test_comparator.cpython-311.pyc
    │   ├── test_db_manager.cpython-311.pyc
    │   ├── test_helpers.cpython-311.pyc
    │   └── test_video_tracker.cpython-311.pyc
    ├── test_comparator.py
    ├── test_db_manager.py
    ├── test_helpers.py
    └── test_video_tracker.py

```
