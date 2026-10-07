import time
import mujoco
import mujoco.viewer
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
XML_PATH = "robotics/mujoco_menagerie/skydio_x2/x2.xml"
HOVER_THRUST = 3.2495625          # from keyframe
MAX_THRUST = 13.0

# ============================================================
# CONTROL GAINS (tune as needed)
# ============================================================
KP_ROLL = 1.5
KP_PITCH = 1.5
KP_YAW = 0.5

KD_ROLL = 0.15
KD_PITCH = 0.15
KD_YAW = 0.05

# ============================================================
# KEYBOARD COMMAND MAGNITUDES
# ============================================================
THROTTLE_STEP = 0.05
MAX_ROLL = np.deg2rad(10)
MAX_PITCH = np.deg2rad(10)
YAW_STEP = np.deg2rad(5)

# ============================================================
# LOAD MODEL
# ============================================================
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Start at the official hover keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)

# ============================================================
# MOTOR IDS
# ============================================================
motor_names = ["thrust1", "thrust2", "thrust3", "thrust4"]
motor_ids = []
for name in motor_names:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id == -1:
        raise RuntimeError(f"Actuator '{name}' not found")
    motor_ids.append(actuator_id)

print("==========================================")
print("       SKYDIO X2 STABILIZED CONTROL")
print("==========================================")
for name, actuator_id in zip(motor_names, motor_ids):
    print(f"{name}: actuator {actuator_id}")
print()
print("W / S       Increase / decrease thrust")
print("A / D       Roll left / right")
print("UP / DOWN   Pitch forward / backward")
print("Q / E       Yaw")
print("SPACE       Level + hover thrust")
print("==========================================")

# ============================================================
# COMMAND VARIABLES
# ============================================================
throttle = HOVER_THRUST
desired_roll = 0.0
desired_pitch = 0.0
desired_yaw = 0.0

# ============================================================
# KEYBOARD (set-and-hold attitude; SPACE levels)
# ============================================================
def key_callback(keycode):
    global throttle, desired_roll, desired_pitch, desired_yaw

    if keycode in (ord("w"), ord("W")):
        throttle = min(throttle + THROTTLE_STEP, MAX_THRUST)
    elif keycode in (ord("s"), ord("S")):
        throttle = max(throttle - THROTTLE_STEP, 0.0)

    elif keycode in (ord("a"), ord("A")):
        desired_roll = -MAX_ROLL
    elif keycode in (ord("d"), ord("D")):
        desired_roll = MAX_ROLL

    elif keycode == 265:          # UP
        desired_pitch = MAX_PITCH
    elif keycode == 264:          # DOWN
        desired_pitch = -MAX_PITCH

    elif keycode in (ord("q"), ord("Q")):
        desired_yaw += YAW_STEP
    elif keycode in (ord("e"), ord("E")):
        desired_yaw -= YAW_STEP

    elif keycode == 32:           # SPACE
        throttle = HOVER_THRUST
        desired_roll = 0.0
        desired_pitch = 0.0
        desired_yaw = 0.0

# ============================================================
# ORIENTATION HELPERS
# ============================================================
def get_body_orientation():
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x2")
    return data.xquat[body_id].copy()

def quaternion_to_euler(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw

# ============================================================
# MAIN SIMULATION
# ============================================================
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        # Current orientation
        q = get_body_orientation()
        roll, pitch, yaw = quaternion_to_euler(q)

        # Body-frame angular rates (free-joint convention)
        # data.qvel[0:3] = linear (world), data.qvel[3:6] = angular (body)
        roll_rate, pitch_rate, yaw_rate = data.qvel[3:6]

        # Attitude errors
        roll_error = desired_roll - roll
        pitch_error = desired_pitch - pitch
        yaw_error = desired_yaw - yaw

        # PD attitude controller
        roll_control = KP_ROLL * roll_error - KD_ROLL * roll_rate
        pitch_control = KP_PITCH * pitch_error - KD_PITCH * pitch_rate
        yaw_control = KP_YAW * yaw_error - KD_YAW * yaw_rate

        # Correct X-configuration mixer for Skydio X2 layout
        # thrust1 (RR, -x -y, gear_yaw=-0.0201)
        # thrust2 (RL, -x +y, gear_yaw=+0.0201)
        # thrust3 (FL, +x +y, gear_yaw=-0.0201)
        # thrust4 (FR, +x -y, gear_yaw=+0.0201)
        motor1 = throttle + pitch_control - roll_control - yaw_control
        motor2 = throttle + pitch_control + roll_control + yaw_control
        motor3 = throttle - pitch_control + roll_control - yaw_control
        motor4 = throttle - pitch_control - roll_control + yaw_control

        controls = np.clip(
            np.array([motor1, motor2, motor3, motor4]),
            0.0,
            MAX_THRUST
        )

        for actuator_id, control in zip(motor_ids, controls):
            data.ctrl[actuator_id] = control

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
