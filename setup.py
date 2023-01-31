from setuptools import setup, find_packages
from larvapicker.config.__version__ import __version__

requirements = [
    'docopt',
    'ffmpeg',
    'h5py',
    'json5',
    'matplotlib',
    'numpy',
    'opencv-python',
    'pathlib',
    'pandas',
    'pyspin',
    'pyzmq',
    'scipy',
    'scikit-learn',
    'scikit-image',
    'pyserial',
    'torch'
]

setup(
    name='larvapicker',
    version=__version__,
    description='Drivers and analysis software for the Larva Picker Robot.',
    author='James Yu, Vivek Venkatachalam',
    author_email='yu.hyo@northeastern.edu',
    url='https://github.com/venkatachalamlab/LarvaPickerRobot',
    entry_points={
        'console_scripts': [
            'calibrator=larvapicker.devices.calibrator:main',
            'camera=larvapicker.devices.camera:main',
            'larva_picker=larvapicker.devices.larva_picker:main',
            'logger=larvapicker.devices.logger:main',
            'robot=larvapicker.devices.robot:main',
            'port=larvapicker.analysis_tools.utils.port:main',
            'parser=larvapicker.analysis_tools.parser.main:main',
            'posture_tracker=larvapicker.analysis_tools.posture_tracker.main:main',
            'pt_flagger=larvapicker.analysis_tools.posture_tracker.flag:main',
            'pt_trainer=larvapicker.analysis_tools.posture_tracker.train:main',
            'state_tracker=larvapicker.analysis_tools.state_tracker.main:main',
            'st_flagger=larvapicker.analysis_tools.state_tracker.flag:main',
            'st_trainer=larvapicker.analysis_tools.state_tracker.train:main',
            'compiler=larvapicker.analysis_tools.compiler.main:main'
        ],
    },
    keywords=['behavior', 'drosophila', 'robotics'],
    # install_requires=requirements,
    packages=find_packages()
)
