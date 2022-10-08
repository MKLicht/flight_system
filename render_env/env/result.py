from math import pi, cos, sin, acos

import numpy as np

lock_angle = 30
lock_dist_min = 100
lock_dist_max = 5000


def is_out(x, y, z):
    return not 500 < z < 19500


def get_angle(dx, dy, dz, yaw, pitch):
    a = np.array([dx, dy, dz])
    b = np.array([cos(yaw) * cos(pitch), sin(yaw) * cos(pitch), sin(pitch)])
    return acos(a.dot(b) / (1e-6 + np.linalg.norm(a) * np.linalg.norm(b)))


def get_result(state_dict):
    for i, fighter in enumerate(state_dict['fighters']):
        if fighter['dead']:
            continue
        x, y, z = fighter['x'], fighter['y'], fighter['z']
        fighter['out'] = is_out(x, y, z)
        pitch, yaw = fighter['pitch'] * pi / 180, fighter['yaw'] * pi / 180

        for oi, ofighter in enumerate(state_dict['fighters']):
            if ofighter['dead'] or (ofighter['is_red'] == fighter['is_red']):
                continue
            dx, dy, dz = ofighter['x'] - x, ofighter['y'] - y, ofighter['z'] - z
            dist = np.linalg.norm([dx, dy, dz])
            angle = get_angle(dx, dy, dz, yaw, pitch) * 180 / pi
            fighter['angle1'].append({
                'id': oi,
                'angle': angle
            })
            ofighter['angle2'].append({
                'id': i,
                'angle': angle
            })
            if angle < lock_angle and lock_dist_min < dist < lock_dist_max:
                fighter['lock'] += 1
                ofighter['locked'] += 1
