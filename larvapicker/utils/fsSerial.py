# Copyright (c) 2016, FlySorter LLC

import sys
import glob
import serial
import time


def coerce_to_bytes(x):
    if isinstance(x, bytes):
        return x
    else:
        return bytes(x, "ascii")

# Serial communications class that is used for multiple devices.


class fsSerial:
    """Serial class for generic serial device."""

    WaitTimeout = 3
    portName = ""

    def __init__(self, port, baud=9600, timeout=float(0.1)):
        self.isOpened = False

        while True:
            try:
                self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
                print('Opened port:', port)
                break
            except:
                print("******* FAILED to open port:", port)
        self.isOpened = True

    def close(self):
        self.ser.close()

    # Retrieve any waiting data on the port
    def get_ser_output(self):
        # print ("GSO:")
        output = b''
        while True:
            # read() blocks for the timeout set above *if* there is nothing to read
            #   otherwise it returns immediately
            byte = self.ser.read(1)
            if byte is None or byte == b'':
                break
            output += byte
            if byte == b'\n':
                break
        # print ("GSO Output:", output)
        return output

    # Block and wait for the device to reply with "ok" or "OK"
    # Times out after self.WaitTimeout (set above)
    def wait_for_ok(self):
        # print ("WFO:")

        # Certain serial errors can be fixed by resetting
        reset_trig = False

        output = b''
        timeout_max = self.WaitTimeout / self.ser.timeout
        timeout_count = 0
        while True:
            byte = self.ser.read(1)
            if byte is None or byte == b'':
                timeout_count += 1
                time.sleep(1)
            else:
                output += byte
            if timeout_count > timeout_max:
                print('Serial timeout.')
                break
            if byte == b'\n':
                break
        # print ("WFO Output:", output)
        output = output.decode("ascii")

        # Checks for any errors
        if not output.startswith("ok") and not output.startswith("OK"):
            print("Unexpected serial output:", output.rstrip('\r\n'), "(", ''.join(x for x in output), ")")
            reset_trig = True

        return reset_trig

    # Send a command to the device via serial port
    # Asynchronous by default - doesn't wait for reply
    def send_cmd(self, cmd):
        # print ("SC:", cmd)
        self.ser.write(bytes(cmd, "ascii"))
        self.ser.flush()

    # Send a command to the device via serial port
    # Waits to receive reply of "ok" or "OK" via waitForOK()
    def send_sync_cmd(self, cmd):
        # print ("SSC:", cmd)
        self.ser.flushInput()
        self.ser.write(bytes(cmd, "ascii"))
        self.ser.flush()
        reset_trig = self.wait_for_ok()

        return reset_trig

    # Send a command and retrieve the reply
    def send_cmd_get_reply(self, cmd):
        self.ser.flushInput()
        self.ser.write(bytes(cmd, "ascii"))
        self.ser.flush()
        return self.get_ser_output().decode("ascii")


def list_available_ports():
    """Lists serial ports"""

    if sys.platform.startswith('win'):
        ports = ['COM' + str(i + 1) for i in range(64)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this is to exclude your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass

    return result


def find_dispenser():
    port_list = list_available_ports()

    for port in port_list:
        s = fsSerial(port, 9600)
        s.send_cmd('V')
        time.sleep(0.25)
        r = s.get_ser_output()

        if r.startswith("  V"):
            # print ("Port:", port, "is first dispenser found")
            dispenser = s
            dispenser.ser.flushInput()
            return dispenser

        s.ser.flushInput()
        s.ser.flushOutput()
        s.close()

    return None


def find_smoothie():
    port_list = list_available_ports()

    for port in port_list:
        s = fsSerial(port, 115200)
        print(s)
        r = s.send_cmd_get_reply('version\n')
        print("Reply: ", r)
        if r.startswith("Build version:"):
            print("Port: {} is first Smoothie found".format(port))
            smoothie = s
            smoothie.ser.flushInput()
            return smoothie
        s.ser.flushInput()
        s.ser.flushOutput()
        s.close()

    return None
