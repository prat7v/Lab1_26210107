
import time
import mujoco
import mujoco.viewer

# ============================================================
# CONFIGURATION
# ============================================================

XML_PATH = (
    "robotis_mujoco_menagerie/robotis_tb3/"
    "scene_turtlebot3_waffle_pi.xml"
)

# Wheel velocity command [rad/s]
MAX_SPEED = 5.0

# ============================================================
# LOAD MUJOCO MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# ============================================================
# FIND ACTUATORS BY NAME
# ============================================================

left_motor = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "wheel_left"
)

right_motor = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "wheel_right"
)

print("Left actuator ID :", left_motor)
print("Right actuator ID:", right_motor)

# ============================================================
# KEYBOARD CONTROL
# ============================================================

# Current commands
left_velocity = 0.0
right_velocity = 0.0


def key_callback(keycode):
    global left_velocity, right_velocity

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    if keycode == ord("W") or keycode == ord("w"):
        left_velocity = MAX_SPEED
        right_velocity = MAX_SPEED

    # --------------------------------------------------------
    # Reverse
    # --------------------------------------------------------
    elif keycode == ord("S") or keycode == ord("s"):
        left_velocity = -MAX_SPEED
        right_velocity = -MAX_SPEED

    # --------------------------------------------------------
    # Turn left
    # --------------------------------------------------------
    elif keycode == ord("A") or keycode == ord("a"):
        left_velocity = -MAX_SPEED
        right_velocity = MAX_SPEED

    # --------------------------------------------------------
    # Turn right
    # --------------------------------------------------------
    elif keycode == ord("D") or keycode == ord("d"):
        left_velocity = MAX_SPEED
        right_velocity = -MAX_SPEED

    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------
    elif keycode == 32:  # SPACE
        left_velocity = 0.0
        right_velocity = 0.0


# ============================================================
# RUN SIMULATION
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    print("\n========================================")
    print("      TURTLEBOT3 WAFFLE PI CONTROL")
    print("========================================")
    print("W     -> Forward")
    print("S     -> Reverse")
    print("A     -> Turn Left")
    print("D     -> Turn Right")
    print("SPACE -> Stop")
    print("ESC   -> Close viewer")
    print("========================================\n")

    while viewer.is_running():

        # ----------------------------------------------------
        # Apply wheel velocities
        # ----------------------------------------------------

        data.ctrl[left_motor] = left_velocity
        data.ctrl[right_motor] = right_velocity

        # ----------------------------------------------------
        # Step simulation
        # ----------------------------------------------------

        mujoco.mj_step(model, data)

        # ----------------------------------------------------
        # Synchronize viewer
        # ----------------------------------------------------

        viewer.sync()

        # Maintain approximately real-time simulation
        time.sleep(model.opt.timestep)
