# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP2
"""

import math
import numpy as np
from alien import Alien


# 计算点到线段的最短距离
def point_to_segments(x, y, x1, y1, x2, y2):
    cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)  # 矢量乘
    if cross <= 0:
        return math.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1))
    d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    if cross >= d2:
        return math.sqrt((x - x2) * (x - x2) + (y - y2) * (y - y2))
    r = cross / d2
    px = x1 + (x2 - x1) * r
    py = y1 + (y2 - y1) * r
    return math.sqrt((x - px) * (x - px) + (y - py) * (y - py))


# 判断两条线段是否相交
def cross_segments(p1, p2, p3, p4):
    if p4[0] - p3[0] == 0:
        if (p2[0] - p1[0] == 0) & ((max(p2[1], p1[1]) < min(p3[1], p4[1])) | (min(p2[1], p1[1]) > max(p3[1], p4[1]))):
            return False
    if p4[1] - p3[1] == 0:
        if (p2[1] - p1[1] == 0) & ((max(p2[0], p1[0]) < min(p3[0], p4[0])) | (min(p2[0], p1[0]) > max(p3[0], p4[0]))):
            return False

    v1 = (p3[0] - p1[0], p3[1] - p1[1])
    v2 = (p2[0] - p1[0], p2[1] - p1[1])
    v3 = (p4[0] - p1[0], p4[1] - p1[1])
    cross1 = v1[0] * v2[1] - v1[1] * v2[0]
    cross2 = v3[0] * v2[1] - v3[1] * v2[0]

    v4 = (p2[0] - p3[0], p2[1] - p3[1])
    v5 = (p4[0] - p3[0], p4[1] - p3[1])
    v6 = (p1[0] - p3[0], p1[1] - p3[1])
    cross3 = v4[0] * v5[1] - v5[0] * v4[1]
    cross4 = v6[0] * v5[1] - v5[0] * v6[1]
    return (cross1 * cross2 <= 0) & (cross4 * cross3 <= 0)


def does_alien_touch_wall(alien, walls, granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endx), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """
    keep_distance = granularity / math.sqrt(2)
    alien_P = alien.get_centroid()
    half_width = alien.get_width()
    radium = half_width + keep_distance
    if alien.is_circle():
        for block in walls:
            distance = point_to_segments(alien_P[0], alien_P[1], block[0], block[1], block[2], block[3])
            if (distance < radium) | np.isclose(distance, radium):
                return True

    else:
        real_half_width = half_width + keep_distance
        head = alien.get_head_and_tail()[0]
        tail = alien.get_head_and_tail()[1]
        if alien.get_shape() == 'Horizontal':
            p2 = (head[0], head[1] + real_half_width)
            p1 = (tail[0], tail[1] + real_half_width)
            p4 = (head[0], head[1] - real_half_width)
            p3 = (tail[0], tail[1] - real_half_width)

        if alien.get_shape() == 'Vertical':
            p1 = (tail[0] - real_half_width, tail[1])
            p3 = (tail[0] + real_half_width, tail[1])
            p2 = (head[0] - real_half_width, head[1])
            p4 = (tail[0] + real_half_width, head[1])
        for block in walls:
            distance1 = point_to_segments(head[0], head[1], block[0], block[1], block[2], block[3])
            distance2 = point_to_segments(tail[0], tail[1], block[0], block[1], block[2], block[3])

            if (distance2 < radium) | np.isclose(distance2, radium):
                return True
            if (distance1 < radium) | np.isclose(distance1, radium):
                return True
            if cross_segments(p1, p2, (block[0], block[1]), (block[2], block[3])):
                return True
            if cross_segments(p3, p4, (block[0], block[1]), (block[2], block[3])):
                return True

    return False


def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal
        
        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be multiple goals
        
        Return:
            True if a goal is touched, False if not.
    """
    head = alien.get_head_and_tail()[0]
    tail = alien.get_head_and_tail()[1]
    width = alien.get_width()
    for goal in goals:
        distance = point_to_segments(goal[0],goal[1],head[0],head[1],tail[0],tail[1])
        if (distance<width+goal[2])| np.isclose(distance,width+goal[2]):
            return True
    return False


def is_alien_within_window(alien, window, granularity):
    """Determine whether the alien stays within the window
        
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """
    keep_distance = granularity / math.sqrt(2)
    alien_P = alien.get_centroid()
    real_halwidth = alien.get_width() + keep_distance
    distance1 = alien_P[0]
    distance2 = alien_P[1]
    distance3 = window[0] - distance1
    distance4 = window[1] - distance2
    if alien.is_circle():
        temp = min(distance3, distance4, distance1, distance2)
        return temp > real_halwidth
    temp_long = min(distance1, distance3)
    temp_width = min(distance4, distance2)
    if alien.get_shape() == 'Horizontal':
        A = temp_long >(real_halwidth + alien.get_length() / 2)
        B = temp_width >real_halwidth
        return A & B
    if alien.get_shape() == 'Vertical':
        A = temp_width > (real_halwidth + alien.get_length() / 2)
        B = temp_long > real_halwidth
        return A & B

    return True


if __name__ == '__main__':
    # Walls, goals, and aliens taken from Test1 map
    walls = [(0, 100, 100, 100),
             (0, 140, 100, 140),
             (100, 100, 140, 110),
             (100, 140, 140, 130),
             (140, 110, 175, 70),
             (140, 130, 200, 130),
             (200, 130, 200, 10),
             (200, 10, 140, 10),
             (175, 70, 140, 70),
             (140, 70, 130, 55),
             (140, 10, 130, 25),
             (130, 55, 90, 55),
             (130, 25, 90, 25),
             (90, 55, 90, 25)]
    goals = [(110, 40, 10)]
    window = (220, 200)

    # Initialize Aliens and perform simple sanity check.
    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    assert not does_alien_touch_wall(alien_ball, walls,
                                     0), f'does_alien_touch_wall(alien, walls) with alien config {alien_ball.get_config()} returns True, expected: False'
    assert not does_alien_touch_goal(alien_ball,
                                     goals), f'does_alien_touch_goal(alien, walls) with alien config {alien_ball.get_config()} returns True, expected: False'
    assert is_alien_within_window(alien_ball, window,
                                  0), f'is_alien_within_window(alien, walls) with alien config {alien_ball.get_config()} returns False, expected: True'

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    assert not does_alien_touch_wall(alien_horz, walls,
                                     0), f'does_alien_touch_wall(alien, walls) with alien config {alien_horz.get_config()} returns True, expected: False'
    assert not does_alien_touch_goal(alien_horz,
                                     goals), f'does_alien_touch_goal(alien, walls) with alien config {alien_horz.get_config()} returns True, expected: False'
    assert is_alien_within_window(alien_horz, window,
                                  0), f'is_alien_within_window(alien, walls) with alien config {alien_horz.get_config()} returns False, expected: True'

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    assert does_alien_touch_wall(alien_vert, walls,
                                 0), f'does_alien_touch_wall(alien, walls) with alien config {alien_vert.get_config()} returns False, expected: True'
    assert not does_alien_touch_goal(alien_vert,
                                     goals), f'does_alien_touch_goal(alien, walls) with alien config {alien_vert.get_config()} returns True, expected: False'
    assert is_alien_within_window(alien_vert, window,
                                  0), f'is_alien_within_window(alien, walls) with alien config {alien_vert.get_config()} returns False, expected: True'

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()
        assert does_alien_touch_wall(alien, walls, 0) == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {not truths[0]}, expected: {truths[0]}'
        assert does_alien_touch_goal(alien, goals) == truths[
            1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {not truths[1]}, expected: {truths[1]}'
        assert is_alien_within_window(alien, window, 0) == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {not truths[2]}, expected: {truths[2]}'


    alien_positions = [
        # Sanity Check
        (0, 100),

        # Testing window boundary checks
        (25.5, 25.5),
        (25.4, 25.4),
        (194.5, 174.5),
        (194.6, 174.6),

        # Testing wall collisions
        (30, 112),
        (30, 113),
        (30, 105.5),
        (30, 105.49),  # Very close edge case
        (30, 135),
        (140, 120),
        (187.5, 70),  # Another very close corner case, right on corner

        # Testing goal collisions
        (110, 40),
        (145.5, 40),  # Horizontal tangent to goal
        (110, 62.5),  # ball tangent to goal

        # Test parallel line oblong line segment and wall
        (50, 100),
        (200, 100),
        (205.5, 100)  # Out of bounds
    ]

    # Truths are a list of tuples that we will compare to function calls in the form (does_alien_touch_wall, does_alien_touch_goal, is_alien_within_window)
    alien_ball_truths = [
        (True, False, False),
        (False, False, True),
        (False, False, True),
        (False, False, True),
        (False, False, True),
        (True, False, True),
        (False, False, True),
        (True, False, True),
        (True, False, True),
        (True, False, True),
        (True, False, True),
        (True, False, True),
        (False, True, True),
        (False, False, True),
        (True, True, True),
        (True, False, True),
        (True, False, True),
        (True, False, True)
    ]
    alien_horz_truths = [
        (True, False, False),
        (False, False, True),
        (False, False, False),
        (False, False, True),
        (False, False, False),
        (False, False, True),
        (False, False, True),
        (True, False, True),
        (True, False, True),
        (True, False, True),
        (False, False, True),
        (True, False, True),
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, False, True),
        (True, False, False),
        (True, False, False)
    ]
    alien_vert_truths = [
        (True, False, False),
        (False, False, True),
        (False, False, False),
        (False, False, True),
        (False, False, False),
        (True, False, True),
        (True, False, True),
        (True, False, True),
        (True, False, True),
        (True, False, True),
        (True, False, True),
        (False, False, True),
        (True, True, True),
        (False, False, True),
        (True, True, True),
        (True, False, True),
        (True, False, True),
        (True, False, True)
    ]

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
