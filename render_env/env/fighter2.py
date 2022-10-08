import math
from math import sin, cos, tan

import numpy as np

PLA_TURN_RATE = 0.1
PLA_LIFT_RATE = 20.0
PLA_ROLL_RATE = 30.0

class Fighter:
    def __init__(self):
        self.old = []

    def __warp(self, h, p, r):
        p %= 360
        h %= 360
        r %= 360
        if (90 < p <= 270):
            h = (h + 180) % 360
            r = (r + 180) % 360
            p = 180 - p 
        if (180 < p <=360):
            p = -(360 - p)
        if (180 < h <=360):
            h = -(360 - h)
        if (180 < r <=360):
            r = -(360 - r)
        return h, p, r
    
    def __dynamics(self, pre_state, action, dt):

        pi = math.acos(-1)
        #r2d = 180.0 / pi
        
        x, y, z = pre_state[1], pre_state[0], pre_state[2]
        heading, pitch, roll = pre_state[3], pre_state[4], pre_state[5]
        V, U, W = pre_state[6], pre_state[7], pre_state[8]
        vt = math.sqrt(abs(V ** 2) + abs(U ** 2) + abs(W ** 2))
        P, Q, R = pre_state[9], pre_state[10], pre_state[11]

        ac_v, ac_p, ac_r, ac_y = action[0], action[1], action[2], action[3]

        state_dot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        ########Euler Angle##########
        if ac_v == 1:
            if U < 800.0:
                U += 50.0 * dt
        elif ac_v == -1:
            if U > 100.0:
                U -= 20.0 * dt
        if ac_p == 1:
            Q = PLA_LIFT_RATE * dt
        elif ac_p == -1:
            Q = -PLA_LIFT_RATE * dt
        if ac_r == 1:
            P = PLA_ROLL_RATE * dt
        elif ac_r == -1:
            P = -PLA_ROLL_RATE * dt
        if ac_y == 1:
            R = PLA_TURN_RATE * dt
        elif ac_y == -1:
            R = -PLA_TURN_RATE * dt

        state_dot[7] = U
        state_dot[9] = P 
        state_dot[10] = Q 
        state_dot[11] = R 

        heading, pitch, roll = pre_state[3], pre_state[4], pre_state[5]
        phi, theta, psi = roll, pitch, -heading
        sphi = sin(phi * pi / 180)
        cphi = cos(phi * pi / 180)
        st = sin(theta * pi / 180)
        ct = cos(theta * pi / 180)
        tt = tan(theta * pi / 180)
        spsi = sin(psi * pi / 180)
        cpsi = cos(psi * pi / 180)
        if (pitch / 90) % 2 != 1:
            phi += P + tt * (Q * sphi + R * cphi)
            theta += Q * cphi - R * sphi
            psi += (Q * sphi + R * cphi) / ct
        else:
            phi += 0
            theta += Q * cphi - R * sphi
            psi += 0
        heading, pitch, roll = self.__warp(-psi, theta, phi)
        state_dot[3] = heading
        state_dot[4] = pitch
        state_dot[5] = roll
    
        ###########new position###############
        V, U, W = pre_state[6], pre_state[7], pre_state[8]
        phi, theta, psi = roll, pitch, -heading
        pi = math.acos(-1)
        sphi = sin(phi * pi / 180)
        cphi = cos(phi * pi / 180)
        st = sin(theta * pi / 180)
        ct = cos(theta * pi / 180)
        spsi = sin(psi * pi / 180)
        cpsi = cos(psi * pi / 180)
        x_dot = U * (ct * spsi) + V * (sphi * spsi * st + cphi * cpsi) + W * (cphi * st * spsi - sphi * cpsi)
        y_dot = U * (ct * cpsi) + V * (sphi * cpsi * st - cphi * spsi) + W * (cphi * st * cpsi + sphi * spsi)
        z_dot = U * st - V * (sphi * ct) - W * (cphi * ct)
        x, y, z = pre_state[1], pre_state[0], pre_state[2]
        state_dot[1] = x + x_dot * dt
        state_dot[0] = y + y_dot * dt
        state_dot[2] = z + z_dot * dt

        if (abs(Q) >= 1e-3):
            state_dot[10] = 0.0

        if (abs(P) >= 1e-3):
            state_dot[9] = 0.0

        if (abs(R) >= 1e-5):
            state_dot[11] = 0.0

        self.old = state_dot

    def reset(self, state):
        x, y, z = state[1], state[0], state[2]
        heading, pitch, roll = state[3], state[4], state[5]
        V, U, W = state[6], state[7], state[8]
        vt = math.sqrt(abs(V ** 2) + abs(U ** 2) + abs(W ** 2))
        P, Q, R = state[9], state[10], -state[11]

        self.old = [x, y, z, heading, pitch, roll, V, U, W, P, Q, R]

    def step(self, dt, action):
        pre_fly = self.old
        self.__dynamics(pre_fly, action, dt)

    def state(self, fighter_dict):
        x, y, z, yaw, pitch, roll, V, U, W, P, Q, R = self.old
        fighter_dict['x'] = x 
        fighter_dict['y'] = y
        fighter_dict['z'] = z
        fighter_dict['v'] = math.sqrt(abs(V ** 2)+abs(U ** 2)+abs(W ** 2))
        fighter_dict['pitch'] = pitch
        fighter_dict['yaw'] = -yaw
        fighter_dict['roll'] = roll