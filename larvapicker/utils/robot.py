import time

from .fsSerial import find_smoothie


class Robot:
    """ Converts common and useful serial inputs into the robot to more intuitive language """

    def __init__(self):
        self.smoothie = find_smoothie()

        while self.smoothie is None:
            EnvironmentError("Smoothieboard not found.")
            print("******* ERROR: Smoothieboard not found.\n")
            self.smoothie = find_smoothie()

        self.reset()
        print("Robot initialized.")

    def close(self):
        self.smoothie.close()

    def reset(self):
        self.smoothie.send_sync_cmd("M999\n")       # Reset the robot
        self.smoothie.send_sync_cmd("M43\n")
        self.smoothie.send_sync_cmd("M45\n")
        self.home()

    def restart(self):
        self.close()
        self.smoothie = None
        while self.smoothie is None:
            self.smoothie = find_smoothie()
        self.reset()
        self.sleep(150)
        self.home()
        print("Robot restarted.")

    def home(self):
        self.set_z(0)
        x, y, z = self.get_position()
        if x > 100:
            x = 25
        if y > 25:
            y = 10
        self.set_xy(x, y)
        self.smoothie.send_sync_cmd("G28\n")        # Home the robot
        self.smoothie.send_sync_cmd("M84\n")        # Unsure what this is for also
        self.smoothie.send_sync_cmd("G90\n")        # Unsure what this is for

    def vacuum_on(self):
        self.smoothie.send_sync_cmd("M42\n")

    def vacuum_off(self):
        self.smoothie.send_sync_cmd("M43\n")

    def air_on(self):
        self.smoothie.send_sync_cmd("M44\n")

    def air_off(self):
        self.smoothie.send_sync_cmd("M45\n")

    def set_z(self, z):
        if z < -22.9:
            print('Robot coordinate out-of-bounds. Set to maximum.')
            z = -22.9
        reset = self.smoothie.send_sync_cmd(f"G01 F2000 Z{z}\n")
        while reset:
            print("Encountered error, resetting, trying again.")
            self.restart()
            reset = self.smoothie.send_sync_cmd(f"G01 F2000 Z{z}\n")

    def set_xy(self, x, y):
        if x < 0:
            print("Robot coordinate out-of-bounds. Set to 0.")
            x = 0
        elif x > 250:
            print("Robot coordinate out-of-bounds. Set to maximum.")
            x = 250
        if y < 0:
            print("Robot coordinate out-of-bounds. Set to 0.")
            y = 0
        elif y > 235:
            print("Robot coordinate out-of-bounds. Set to maximum.")
            y = 234.5

        reset = self.smoothie.send_sync_cmd(f"G01 F9000 X{x} Y{y}\n")
        while reset:
            print("Encountered error, resetting, trying again.")
            self.restart()
            reset = self.smoothie.send_sync_cmd(f"G01 F6000 X{x} Y{y}\n")

    def slow_xy(self, x, y, v):
        if x < 0 or y < 0 or x > 250 or y > 230:
            print("Robot coordinate out-of-bounds. Skipping.")
        else:
            reset = self.smoothie.send_sync_cmd(f"G01 F{v} X{x} Y{y}\n")
            while reset:
                print("Encountered error, resetting, trying again.")
                self.restart()
                reset = self.smoothie.send_sync_cmd(f"G01 F{v} X{x} Y{y}\n")

    def sleep(self, milliseconds):
        self.smoothie.send_sync_cmd(f"G04 P{milliseconds}\n")

    def get_pressure(self):
        return float(self.smoothie.send_cmd_get_reply("M105\n").split(' ')[1].split(':')[1])

    # *** NOTE that if this is called during TRANSIT,
    # it will return the DESTINATION coordinates,
    # NOT current position *** #
    def get_position(self):
        position_string = self.smoothie.send_cmd_get_reply("M114\n")   # position read
        try:
            x = float(position_string.split(' ')[2].split(':')[1])
            y = float(position_string.split(' ')[3].split(':')[1])
            z = float(position_string.split(' ')[4].split(':')[1])
        except:
            print('******* ERROR: invalid response received\n'
                  f'\t\t{position_string}')
            x, y, z = 25, 10, 0
        return [x, y, z]


if __name__ == "__main__":
    r = Robot()
    print("Setting x and y to 20")
    r.set_xy(40, 40)
    time.sleep(2)
    r.close()

