import struct
import numpy as np

class RobotProtocol:
    """
    Standardized memory layout for low-overhead UDP serialization.
    Prevents JSON parsing overhead inside the 100Hz RTOS loop.
    """
    # Format: 6 Floats (Joint Vels) + 1 Float (Gripper) + 1 Integer (Safety State flag)
    PACK_FORMAT = "!7fi" 
    PACK_SIZE = struct.calcsize(PACK_FORMAT)

    @staticmethod
    def serialize_action(joint_velocities: np.ndarray, gripper_pos: float, safe_state: int = 0) -> bytes:
        return struct.pack(
            RobotProtocol.PACK_FORMAT,
            float(joint_velocities[0]), float(joint_velocities[1]), float(joint_velocities[2]),
            float(joint_velocities[3]), float(joint_velocities[4]), float(joint_velocities[5]),
            float(gripper_pos),
            safe_state
        )

    @staticmethod
    def deserialize_action(data: bytes) -> tuple:
        unpacked = struct.unpack(RobotProtocol.PACK_FORMAT, data)
        return np.array(unpacked[:6], dtype=np.float32), unpacked[6], unpacked[7]
