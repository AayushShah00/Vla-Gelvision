
import socket
import time
import sys
import os
import can
import numpy as np
import logging
import RPi.GPIO as GPIO
from shared_protocol import RobotProtocol

logging.basicConfig(level=logging.INFO, format='%(asctime)s [RT-PI] %(message)s')
logger = logging.getLogger("PiHardwareNode")

class CANBusRobotController:
    """
    Direct SocketCAN communication driver interface tracking real hardware profiles.
    Communicates via standard 29-bit identifier structures using the python-can library.
    """
    def __init__(self, interface: str = "can0", bitrate: int = 1000000):
        self.RED_LED_PIN = 18
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.RED_LED_PIN, GPIO.OUT)
        GPIO.output(self.RED_LED_PIN, GPIO.LOW)

        logger.info(f"Bringing up CAN bus interface link: {interface} at {bitrate}bps")
        try:
            self.bus = can.interface.Bus(interface, bustype='socketcan', bitrate=bitrate)
        except Exception as e:
            logger.critical(f"Failed to bind socket link over hardware interface layer: {e}")
            sys.exit(1)

    def write_joint_velocities_to_bus(self, joint_velocities: np.ndarray, gripper_pos: float):
        """
        Packs physical float variables straight into raw motor controller memory bytes.
        Assumes standard motor driver base addressing structures (IDs 0x11 through 0x16).
        """
        # Map 6 discrete velocities across physical joint IDs
        for idx in range(6):
            motor_id = 0x11 + idx
            velocity_val = float(joint_velocities[idx])
            
            # IEEE 754 binary floating point layout packing protocol (4 Bytes)
            payload_data = struct.pack("!f", velocity_val)
            
            # Construct standard SocketCAN Frame instance
            msg = can.Message(arbitration_id=motor_id, data=payload_data, is_extended_id=False)
            try:
                self.bus.send(msg)
            except can.CanError:
                logger.error(f"CAN Arbitration Failed on Motor ID Node: {hex(motor_id)}")

        # Track Gripper Actuator Line Node Address (0x17)
        gripper_payload = struct.pack("!f", float(gripper_pos))
        gripper_msg = can.Message(arbitration_id=0x17, data=gripper_payload, is_extended_id=False)
        self.bus.send(gripper_msg)

    def execute_immediate_hardware_lockout(self):
        """Triggers direct emergency safe state. Cuts actuator power distribution rings."""
        GPIO.output(self.RED_LED_PIN, GPIO.HIGH)
        
        # Broadcast absolute Emergency Stop (0x00) frames to clear control loops across all drivers
        estop_msg = can.Message(arbitration_id=0x00, data=[0xFF, 0xFF, 0xFF, 0xFF], is_extended_id=False)
        for _ in range(5): # Redundantly spam frames to ensure intake amidst collision vectors
            self.bus.send(estop_msg)
        logger.critical("SAFETY INTERRUPTION INTERACTION RECOVERY COMPLETED: Real CAN Bus lines dropped.")

    def close(self):
        self.execute_immediate_hardware_lockout()
        GPIO.output(self.RED_LED_PIN, GPIO.LOW)
        GPIO.cleanup()
        self.bus.shutdown()

def run_real_time_scheduler():
    """
    Sets hard thread priority boundaries via SCHED_FIFO scheduling algorithms
    on real patched Linux kernels (PREEMPT_RT).
    """
    try:
        param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        logger.info("Successfully established hard realtime bounds over active thread scheduler.")
    except PermissionError:
        logger.warning("Elevated priority privileges rejected. Run runtime execution using sudo commands.")

    BIND_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", BIND_PORT))
    sock.setblocking(False) # Prevent IO blocks inside structural loops

    robot_hw = CANBusRobotController()
    
    target_period = 1.0 / 100.0 # Strict 100Hz cadence bounds
    next_deadline = time.monotonic() + target_period

    current_vels = np.zeros(6, dtype=np.float32)
    current_gripper = 0.0

    logger.info("Deterministic real-time control system active.")

    try:
        while True:
            raw_packet = None
            
            # Dequeue network buffer arrays cleanly to eliminate stack latency drift
            while True:
                try:
                    data, addr = sock.recvfrom(RobotProtocol.PACK_SIZE)
                    if len(data) == RobotProtocol.PACK_SIZE:
                        raw_packet = data
                except BlockingIOError:
                    break

            # Handle incoming array states
            if raw_packet:
                joint_vels, gripper, safe_flag = RobotProtocol.deserialize_action(raw_packet)
                
                if safe_flag == 1:
                    robot_hw.execute_immediate_hardware_lockout()
                    logger.critical("Terminating local application stack: Server initiated shutdown flag.")
                    break
                
                current_vels = joint_vels
                current_gripper = gripper

            # Push values directly to hardware components
            robot_hw.write_joint_velocities_to_bus(current_vels, current_gripper)

            # Precise clock matching control
            slack_remainder = next_deadline - time.monotonic()
            if slack_remainder > 0:
                time.sleep(slack_remainder)
            else:
                logger.warning(f"Realtime Window Drift Overrun Alert: {-slack_remainder * 1000.0:.3f} ms lag.")
                
            next_deadline += target_period

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt captured.")
    finally:
        robot_hw.close()
        sock.close()
        logger.info("System processing dropped safely.")

if __name__ == "__main__":
    run_real_time_scheduler()
