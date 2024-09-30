from setuptools import setup, find_packages

setup(
    name='boat_detection_project',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='AI-based boat detection and tracking project.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/boat_detection_project',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'ultralytics',
        'scikit-image',
        'PyYAML',
        'python-dotenv'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
