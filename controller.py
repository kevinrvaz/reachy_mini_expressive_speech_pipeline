import os
import sys

import numpy as np
import pygame
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.joystick.init()

joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
pro_controller = joysticks[0]

# buttons
Z_UP = 11
Z_DOWN = 12
Y_LEFT = 13
Y_RIGHT = 14
X_FORWARD = 9
X_BACKWARD = 4
LEFT_ANTENNA = 0
RIGHT_ANTENNA = 2
BODY_LEFT = 4
BODY_RIGHT = 6
ROLL_LEFT = 10
ROLL_RIGHT = 5
YAW_LEFT = 3
YAW_RIGHT = 0
PITCH_UP = 2
PITCH_DOWN = 1

# clamping values
min_z, max_z = -0.05, 0.05
min_x, max_x = -0.15, 0.15
min_y, max_y = -0.15, 0.15
min_antenna, max_antenna = -2.793, 2.793
min_body, max_body = -2.793, 2.793
min_roll, max_roll = -0.5, 0.5
min_yaw, max_yaw = -3.141, 3.141
min_pitch, max_pitch = -0.5, 0.5
current_z, z_step = 0, 0.005
current_x, x_step = 0, 0.01
current_y, y_step = 0, 0.01
current_left_antenna, current_right_antenna, antenna_step = 0, 0, 10
current_body, body_step = 0, 10
current_roll, roll_step = 0, 5
current_yaw, yaw_step = 0, 10
current_pitch, pitch_step = 0, 5


# def main():
with ReachyMini(
    connection_mode="network", media_backend="no_media", automatic_body_yaw=True
) as mini:
    mini.goto_target(head=create_head_pose(), duration=1, antennas=[0.0, 0.0])
    while True:
        button_pressed = False
        try:
            pygame.event.pump()

            # Head Translation
            # Z
            if pro_controller.get_button(Z_UP):
                current_z += z_step
                if current_z > max_z:
                    current_z = max_z
                button_pressed = True
            if pro_controller.get_button(Z_DOWN):
                current_z -= z_step
                if current_z < min_z:
                    current_z = min_z
                button_pressed = True

            # Y
            if pro_controller.get_button(Y_RIGHT):
                current_y += y_step
                if current_y > max_y:
                    current_y = max_y
                button_pressed = True
            if pro_controller.get_button(Y_LEFT):
                current_y -= y_step
                if current_y < min_y:
                    current_y = min_y
                button_pressed = True

            # X
            if pro_controller.get_button(X_FORWARD):
                current_x += x_step
                if current_x > max_x:
                    current_x = max_x
                button_pressed = True
            if round(pro_controller.get_axis(X_BACKWARD)) == 1:
                current_x -= x_step
                if current_x < min_x:
                    current_x = min_x
                button_pressed = True

            # Head Rotation
            # Left Antenna
            if round(pro_controller.get_axis(LEFT_ANTENNA)) == 1:
                current_left_antenna += np.deg2rad(antenna_step)
                if current_left_antenna > max_antenna:
                    current_left_antenna = max_antenna
                button_pressed = True
            if round(pro_controller.get_axis(LEFT_ANTENNA)) == -1:
                current_left_antenna -= np.deg2rad(antenna_step)
                if current_left_antenna < min_antenna:
                    current_left_antenna = min_antenna
                button_pressed = True

            # Right Antenna
            if round(pro_controller.get_axis(RIGHT_ANTENNA)) == 1:
                current_right_antenna += np.deg2rad(antenna_step)
                if current_right_antenna > max_antenna:
                    current_right_antenna = max_antenna
                button_pressed = True
            if round(pro_controller.get_axis(RIGHT_ANTENNA)) == -1:
                current_right_antenna -= np.deg2rad(antenna_step)
                if current_right_antenna < min_antenna:
                    current_right_antenna = min_antenna
                button_pressed = True

            # Roll
            if round(pro_controller.get_axis(ROLL_RIGHT)) == 1:
                current_roll += np.deg2rad(roll_step)
                if current_roll > max_roll:
                    current_roll = max_roll
                button_pressed = True
            if pro_controller.get_button(ROLL_LEFT):
                current_roll -= np.deg2rad(roll_step)
                if current_roll < min_roll:
                    current_roll = min_roll
                button_pressed = True

            # Yaw
            if pro_controller.get_button(YAW_RIGHT):
                current_yaw += np.deg2rad(yaw_step)
                if current_yaw > max_yaw:
                    current_yaw = max_yaw
                button_pressed = True
            if pro_controller.get_button(YAW_LEFT):
                current_yaw -= np.deg2rad(yaw_step)
                if current_yaw < min_yaw:
                    current_yaw = min_yaw
                button_pressed = True

            # Pitch
            if pro_controller.get_button(PITCH_DOWN):
                current_pitch += np.deg2rad(pitch_step)
                if current_pitch > max_pitch:
                    current_pitch = max_pitch
                button_pressed = True
            if pro_controller.get_button(PITCH_UP):
                current_pitch -= np.deg2rad(pitch_step)
                if current_pitch < min_pitch:
                    current_pitch = min_pitch
                button_pressed = True

            # Body Rotation
            if pro_controller.get_button(BODY_RIGHT):
                current_body += np.deg2rad(body_step)
                if current_body > max_body:
                    current_body = max_body
                button_pressed = True
            if pro_controller.get_button(BODY_LEFT):
                current_body -= np.deg2rad(body_step)
                if current_body < min_body:
                    current_body = min_body
                button_pressed = True

            if button_pressed:
                mini.goto_target(
                    head=create_head_pose(
                        y=current_y,
                        z=current_z,
                        x=current_x,
                        roll=current_roll,
                        yaw=current_yaw,
                        pitch=current_pitch,
                        degrees=False,
                    ),
                    antennas=[current_left_antenna, current_right_antenna],
                    body_yaw=current_body,
                    duration=0.2,
                )
        except KeyboardInterrupt:
            print("Exiting controller loop")
            sys.exit(0)


# main()
# while True:
#     pygame.event.pump()
#     print(pro_controller.get_axis(LEFT_ANTENNA))
