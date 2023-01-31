## About

The Larva Picker Robot is an automated system for handling *Drosophila* larvae for continuous long-term observation.
This repository contains the software for driving the robot as well as the analysis tools for the resulting data.

## Table of Contents
1. [Installation](#installation)
2. [Repository structure](#structure)
3. [Getting started](#start)

---

<a name="installation"></a>
## Installation 

### Dependencies

Make sure that **Python (>=3.6)** and the following packages are installed:
  - dataclasses
  - docopt
  - ffmpeg
  - h5py
  - json5
  - matplotlib
  - numpy
  - opencv-python
  - pandas (==1.4.2)
  - pathlib
  - pyspin
  - scikit-learn
  - scikit-image
  - scipy
  - serial
  - setuptools
  - torch (see [PyTorch.org](https://pytorch.org/get-started/locally/) for instructions on installing with CUDA)
  - tqdm
  - zmq

You may also want the following programs/drivers:
* [SmoothieBoard drivers](http://smoothieware.org/)
* [Pronterface](http://www.pronterface.com/)

### Build from source

1. Clone git repository: 
  ```bash
  git clone https://github.com/venkatachalamlab/LarvaPickerRobot.git
  ```  

2. Navigate to the cloned directory on your local machine.

3. Checkout the current release:
```bash
git checkout v1.0.0
```
Use the following command to see what's new in the most recent release:
```bash
git show v1.0.0
```

4. Install:
  ```bash
  python setup.py install
  ```
  or install in development mode:
  ```bash
  python setup.py develop
  ```

---

<a name="structure"></a>
## Repository Structure 

**parts** -- contains STL's and SolidWorks files for some of the hardware for the robot as well as a parts list.

**devices** -- drivers for main devices or ZMQ nodes

* `calibrator.py` -- acts as a hub between camera, robot, and the user by taking prompts from CLI to determine if the calibration for the robot needs to be updated, codenamed "lp" for the pub-sub servers
* `camera.py` -- controls camera, codenamed "cam" for the pub-sub servers
* `robot.py` -- controls the Larva Picker robot, codenamed "rbt" for the pub-sub servers
* `larva_picker.py` -- acts a hub between camera and robot by processing the images from the camera to determine when and how the robot should be activated to pick up a larva, codenamed "lp" for the pub-sub servers
* `logger.py` -- handles the video recorder and other writers to save pertinent information to disk, codenamed "log" for the pub-sub servers

**methods** -- functions and variables that are either often used by a device or shared across devices

* `images.py` -- image processing tools
* `loops.py` -- key functions to define different operation protocols used by larva_picker.py
* `robot.py` -- key functions used by robot.py

**config** -- static configuration files

* `constants.py` -- mostly path names for all the key files, some fixed variables like camera resolution here as well
* `\_\_version\_\_.py` -- repository version
* `smoothie_config.py` -- serial configuration for the SmoothieBoard

**utils** -- lower-level code for the robot and templates for ZMQ objects

* `fsSerial.py` -- wraps serial commands to natural language commands
* `robot.py` -- wraps common command sequences to intuitive natural language commands
* `poller.py` -- template for ZMQ poller object
* `publisher.py` -- template for ZMQ publisher object
* `subscriber.py` -- template for ZMQ subscriber object

**analysis_tools** -- tools for data analysis
* **parser** -- image processing tool for detecting, identifying, and separating individual larvae
    * `main.py` -- runs parser
    * `larva.py` -- wraps writers and mutable properties for each individual larva
* **posture_tracker** -- neural network for determining larval posture (outputs head, tail, midspine for each larva in each video frame)
    * `main.py` -- runs posture_tracker
    * `model.py` -- defines layer sequence for the posture_tracker neural network model
    * `flag.py` -- launches UI for producing ground-truth posture data for training posture_tracker
    * `train.py` -- runs training algorithm for posture_tracker
* **state_tracker** -- neural network for classifying larval behavioral state at each time point
    * `main.py` -- runs state_tracker
    * `model.py` -- defines layer sequence for the state_tracker neural network model
    * `flag.py` -- launches UI for producing ground-truth posture data for training state_tracker
    * `train.py` -- runs training algorithm for state_tracker
* **compiler** -- combines results from parser, posture_tracker, and state_tracker across multiple animals to quantify useful behavioral metrics, such as crawl speed, turn rate, thermotaxis index, etc.
    * `main.py` -- runs compiler
* **utils** -- common functions used across analysis_tools
    * `conv_modules.py` -- wrappers for convolution layers, including a convolutional LSTM cell
    * `io.py` -- templates for writers and IO functions
    * `streamers.py` -- PyTorch dataloaders for posture_tracker and state_tracker

---

<a name="start"></a>
## Getting Started 

For instructions on how to operate the robot, see [this ReadMe](https://github.com/venkatachalamlab/LarvaPickerRobot/blob/main/larvapicker/ReadMe.md).

For instructions on how to analyze the data taken with the robot, see [this ReadMe](https://github.com/venkatachalamlab/LarvaPickerRobot/blob/main/larvapicker/analysis_tools/ReadMe.md).

