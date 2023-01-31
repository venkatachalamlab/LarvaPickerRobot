## About

The Larva Picker is an automated system for handling *Drosophila* larvae for continuous long-term observation.
Follow this ReadMe for instructions on how to operate the robot.

## Table of Contents
1. [Getting started](#start)
2. [Robot calibration](#calibration)
3. [Run larva picker](#run)
4. [Troubleshooting](#troubleshooting)

---

<a name="start"></a>
## Getting Started 

Test connection to robot using Pronterface. 
The port the board is using can be checked in the Device Manager and the baudrate is 19,200. Should be able to home robot, then drive in X/Y/Z (positive directions in X/Y to start, negative in Z).

If the robot gives "print paused @__": use command M999. ***ALWAYS RE-HOME ROBOT AFTER THIS COMMAND***
You can also open solenoid valves with M42 and M44 commands (type into Pronterface). M43 / M45 close valves. M105 reads the pressure sensor. 
You can add buttons in Pronterface for those actions. This can be done by using the plus button on the bottom of the grid.

Pressure is controlled by four valves, the air and vacuum valves directly from the source, and the two small valves on robot.
Adjust flow control valves, shooting for a reading of ~50 when vacuum is open, and, ~20 for air.

### Prior to setting up the robot: 
    Pour a large 22cm-by-22cm agar arena.
 
    Make sure that:
    
    The Ethernet cord is connected to the camera
    The Smoothie board is connected to the computer
    The SIM Card is FULLY inserted on the hardware
    The Robot plugged into wall
    The room lights are OFF (It may stay ON during the calibration)
    The LED lights are ON

---

<a name="calibration"></a>
## Calibrating the robot 

      Before continuing:
      
      **Make sure Vacuum is ON, and Air is OFF!**
      Don't be shy about running the calibrations more than once if anything at all goes wrong or seems off!

**1. Make sure Larva Picker software is properly installed onto your environment.**

**2. RUN `camera.py`**

Use the following command:
```bash
camera
```

**3. RUN `robot.py`**

Use the following command:
```bash
robot
```

* Make sure you can see the camera view window and hear the robot booting up.

**3. Check Air and Vacuum pressure**

* When you first boot up `robot.py`, her CLI will prompt you to set up the vacuum and air pressures, respectively. Input "y" or "Y" once the pressure is at a good place (~40 when vacuum is open, and, ~10 for air). Each time you input "n", it will give you the current pressure reading.

* Air and Vacuum can be adjusted in 2 ways:
    * Coarse adjustment from source
    * Fine adjustement from the small knobs attached the the robot
        * Vacuum should be between 30-40, but no more than 52.7
        * Air should be around 10
    
* Wait until you receive the `READY FOR ACTION` message before proceeding.

**4. RUN `calibrator.py`**

Use the following command:
```bash
calibrator
```

**5. Follow the instructions on the CLI for robot calibration.**

* First, the program will ask if you want to update the robot-camera homography, which defines the transformation between the camera or image coordinates to robot internal coordinates.
        * Place a calibration pattern (checkerboard mat works well) on the base plate, making sure the design covers the field of view of the camera
	* Input (type then Enter) yes (or just "y"), the camera will take a picture of the calibration design. The program uses the image to detect corners in the calibration design and mark them on the image.
	* Look at the image that pops up and check that there are enough corners detected (shoot for at least 12 corners), and that there are no false corners (like along the edges of a square) that have been detected as well.
	* If either of these conditions are not met, press "e" in the image window. This will prompt the program to retake the image.
	* If there are issues with the corner detections, it's usually due to dust reflecting light against a black background. You can brush off any obvious particles or wipe it with a wet paper towel. In more extreme cases, you may want to adjust the camera settings like the aperture and exposure.
	* Once satisfied with the corners detected, press "esc" to continue.
	* Now the robot will start and move to each corner that was detected by the camera.
	* At each calibration point, use the WASD keys on the “Robot controller” window to adjust the robot position until it is directly above the corner. 
		* Sometimes the feature detection will slightly be off the corner -- the "Robot controller" window will show you a tiny red dot inside a larger red circle on top of the calibration design for the current calibration point. Try to aim the robot to be directly above the dot. The red circle is approximately the same width as the nozzle tip, so use that to your advantage as well.
		* Press "q" to move onto the next calibration point.
	* Once finished, you will be prompted to save them to the homography file in the *robot’s CLI* (NOT calibrator’s) - input "y" to save.
	* You will be prompted to check calibration. You can reperform the calibration to fine-tune the homography until little to no adjustments are needed during calibration.
	* Remove the calibration design when finished.

#### You should not have to update the homography every single time. If you find that the robot is consistently hitting the correct target during calibration, you can move on. However, the height map DOES need to be recalibrated EVERY time!

* Back on the calibrator’s CLI, the program will ask if you want to update the height calibration.
	* **PLACE THE AGAR ARENA ON THE MOAT PLATE AND FILL THE MOAT WITH WATER BEFORE CONTINUING!** 
	* It's also a good idea to wait a few minutes for the agar to settle and equilibriate with the water. This becomes critical if agar is being reused and was in storage in the fridge prior.
	* Once you input "y", the robot will start and move to each point on the calibration grid automatically and the nozzle moves down towards the agar until it experiences a pressure change when the vaccuuming tip of the probe gets closed off on contact with the agar, leading to no air entering the nozzle. It will read the pressure and record the height at that point and move onto the next calibration point. It will home between each row.
	* Once finished, in *robot’s CLI*, the program will ask if you want to save - input "y" to save.

* The calibrator will take a quick background image for image processing during the experiment and exit out. The camera and robot should still be running.

---  

<a name="run"></a>
## Running Larva Picker 

    Prior to picking larva, collect larva from fly vials


**6. RUN `logger.py`**

Use the following command:
```bash
logger
```

* Follow the input prompts to add useful notes for the robot activity log (`log.json`) and metadata (`metadata.json`).

* Otherwise the experiment will not be recorded in any way! Good for testing out hardware or software prototypes, but not if you were planning to use the data.

**7. Set up the experiment arena** 

* Place larvae at the center of the agar bed.

* Fill the water moat with water using the IV system. 
     * You can roll up the roll clamp and run the end around the perimeter of the moatuntil the moat is filled and the water evenly distributed. After the initial fill, you can leave the end on the holder at the top right corner of the moat. You can adjust the roll clamp to provide a steady supply of water, or refill as needed.

**8. RUN `larva_picker.py`** 

Use the following command:
```bash
larva_picker
```

* The program will give you an input prompt for the instar stage. This will influence the size of the larva the process will attempt to detect.

* larva_picker’s CLI will continuously read the images as it is updated by the camera. Check that it is finding the correct number of larvae.

* The correct order of events for picking a larva should be:

	* Identifies larva on perimeter and marks it with a blue rectangle on the image window and alerts you that the robot is being activated with a dramatic CLI alert. [“get_water”]
  	* Moves to closest water moat position and dips into water.
  	* Moves out of the way of the camera if necessary. [“crop_image”]
  	* Refreshes image around the previous larva position (currently a 300x300 box). [“search_cropped_image”]
  	* Attempts to pick up the larva. [“pick_larva”]
	* Moves out of the way of the camera.
	* Refreshes image around the pick up position.
	* If the pick was successful, moves to the middle of the agar and drops off larva. [“drop_larva”]
		* If the pick was unsuccessful, it dips into water and tries again.
	* Moves out of the way of the camera to the left edge. [“search_new_image”]
	* Camera refreshes image around the entire perimeter for any more larva or missed catches.
		* Repeats pick-up/drop-off process if any are detected.
   * If no larva are detected in the perimeter, moves back home and prepares for next larva.
    * Moves back home and prepares for next larva.

* Exit program by pressing "esc" while in the live image feed (The "Image Capture" window). This will cleanly end all processes, including the camera and the robot. [“destroy”]

* Note that there is a video (`movie.mp4`) being written in the dataset folder (which uses the current date), but only while the robot is in STANDBY mode, as well as a timestamped compressed file of the image data (`data.h5`). There is also a log text file (`log.json`) that records detected larva positions during STANDBY and the time stamps for when robot was ACTIVE in units of seconds.
  
---

<a name="troubleshooting"></a>
## Troubleshooting common errors

Every step of these programs can have errors- read this before panicking. 

* If camera fails to open when running `camera.py`:
  * Check to make sure ethernet cord is connected 

* If robot is NOT responding AT ALL:
  * Check to make sure smoothie board is connected, it is plugged into the wall, AND that the sim card on the smoothieboard hardware is fully inserted 

* Troubleshooting connection issues:
  * Device manager is your friend, check to make sure smoothieboard is running correctly on the correct port
  * A “G-drive” should open inside your finder when the smoothieboard is running properly, if this drive is not present it may be an issue

* If Pronterface pauses, is unresponsive, is taking too long:
  * M999 resets the robot and Pronterface- **HOME ROBOT AFTER YOU DO THIS**

* If you receive an error message saying “Pressure read too low, is pump/stopcock open?”:
  * Make sure that vacuum and air pressure is correct by using pronterface
  * Make sure vacuum is on and air is off before calibrating 
  * Retype the command: sometimes this error is because of lag, and needs to be retyped a few times before it works 
  
* If robot is not picking up larva:
	* **ADJUST THE VACUUM PRESSURE WITH THE MANUAL VALVES IF IT IS HAVING A HARD TIME PICKING UP LARVA (NEEDS MORE VACUUM) OR HAVING A HARD TIME EJECTING IT OUT OF THE NOZZLE WITH AIR (NEEDS LESS VACUUM, or MORE AIR PRESSURE)**
  * Determine at which point there is an issue
    * If it does not move to the correct position:
      * The XY coordinates are off and you should recalibrate the homography.
    * If it does not move down far enough to reach the larva:
      * The Z coordinates are wrong and you should recalibrate the height map.
    * If it is not putting the larva down after picking it up:
      * The water is **CRUCIAL** to this- check to make sure moat is appropriately filled, and that nozzle reaches the water correctly 
    * If motor stalls, makes a funny noise, and does not return to home:
      * First see if robot's automatic reset protocol functions normally by pressing "SPACE" on the camera window. Otherwise, robot's process will exit.
      * Relaunch `robot.py` and press "SPACE" on the camera window

