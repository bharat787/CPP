"""
Grid based sweep planner

The program takes (x,y) coordinates 
of the corner point of the survey area as 
input, calculates the path and stores the
waypoints in a csv file.

The csv file can be used as input to
matlab code to calculate the energy requirements
of the drone to cover the path generated.
"""
import csv
import math
from enum import IntEnum
from shapely import geometry
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from Mapping.grid_map_lib.grid_map_lib import GridMap
import matplotlib.pyplot as plt

do_animation = True


class SweepSearcher:
    class SweepDirection(IntEnum):
        UP = 1
        DOWN = -1

    class MovingDirection(IntEnum):
        RIGHT = 1
        LEFT = -1

    def __init__(self,
                 moving_direction, sweep_direction, x_inds_goal_y, goal_y):
        self.moving_direction = moving_direction
        self.sweep_direction = sweep_direction
        self.turing_window = []
        self.update_turning_window()
        self.x_indexes_goal_y = x_inds_goal_y
        self.goal_y = goal_y

    def move_target_grid(self, c_x_index, c_y_index, grid_map):
        n_x_index = self.moving_direction + c_x_index
        n_y_index = c_y_index

        # found safe grid
        if not grid_map.check_occupied_from_xy_index(n_x_index, n_y_index,
                                                     occupied_val=0.5):
            return n_x_index, n_y_index
        else:  # occupied
            next_c_x_index, next_c_y_index = self.find_safe_turning_grid(
                c_x_index, c_y_index, grid_map)
            if (next_c_x_index is None) and (next_c_y_index is None):
                # moving backward
                next_c_x_index = -self.moving_direction + c_x_index
                next_c_y_index = c_y_index
                if grid_map.check_occupied_from_xy_index(next_c_x_index,
                                                         next_c_y_index):
                    # moved backward, but the grid is occupied by obstacle
                    return None, None
            else:
                # keep moving until end
                ## go till certain end

                while not grid_map.check_occupied_from_xy_index(
                        next_c_x_index + self.moving_direction,
                        next_c_y_index, occupied_val=0.5):
                    next_c_x_index += self.moving_direction
                self.swap_moving_direction()
            return next_c_x_index, next_c_y_index

    def find_safe_turning_grid(self, c_x_index, c_y_index, grid_map):

        for (d_x_ind, d_y_ind) in self.turing_window:

            next_x_ind = d_x_ind + c_x_index
            next_y_ind = d_y_ind + c_y_index

            # found safe grid
            if not grid_map.check_occupied_from_xy_index(next_x_ind,
                                                         next_y_ind,
                                                         occupied_val=0.5):
                return next_x_ind, next_y_ind

        return None, None

    def is_search_done(self, grid_map):

        for ix in self.x_indexes_goal_y:
            if not grid_map.check_occupied_from_xy_index(ix, self.goal_y,
                                                         occupied_val=0.5):
                return False

        # all lower grid is occupied
        print("exiting")
        return True

    def update_turning_window(self):
        # turning window definition
        # robot can move grid based on it.

        self.turing_window = [
            (self.moving_direction, 0.0),
            (self.moving_direction, self.sweep_direction),
            (0, self.sweep_direction),
            (-self.moving_direction, self.sweep_direction),
        ]

    def swap_moving_direction(self):

        self.moving_direction *= -1
        self.update_turning_window()

    def search_start_grid(self, grid_map):

        x_inds = []
        y_ind = 0
        if self.sweep_direction == self.SweepDirection.DOWN:
            x_inds, y_ind = search_free_grid_index_at_edge_y(
                grid_map, from_upper=True)
        elif self.sweep_direction == self.SweepDirection.UP:
            x_inds, y_ind = search_free_grid_index_at_edge_y(
                grid_map, from_upper=False)

        if self.moving_direction == self.MovingDirection.RIGHT:
            return min(x_inds), y_ind
        elif self.moving_direction == self.MovingDirection.LEFT:
            return max(x_inds), y_ind

        raise ValueError("self.moving direction is invalid ")


def find_sweep_direction_and_start_position(ox, oy):

    # find sweep_direction
    max_dist = 0.0
    vec = [0.0, 0.0]
    sweep_start_pos = [0.0, 0.0]
    for i in range(len(ox) - 1):
        dx = ox[i + 1] - ox[i]
        dy = oy[i + 1] - oy[i]
        d = np.hypot(dx, dy)

        if d > max_dist:
            max_dist = d
            vec = [dx, dy]
            sweep_start_pos = [ox[i], oy[i]]

    return vec, sweep_start_pos


def convert_grid_coordinate(ox, oy, sweep_vec, sweep_start_position):

    tx = [ix - sweep_start_position[0] for ix in ox]
    ty = [iy - sweep_start_position[1] for iy in oy]
    th = math.atan2(sweep_vec[1], sweep_vec[0])
    rot = Rot.from_euler('z', th).as_matrix()[0:2, 0:2]
    converted_xy = np.stack([tx, ty]).T @ rot

    return converted_xy[:, 0], converted_xy[:, 1]


def convert_global_coordinate(x, y, sweep_vec, sweep_start_position):

    th = math.atan2(sweep_vec[1], sweep_vec[0])
    rot = Rot.from_euler('z', -th).as_matrix()[0:2, 0:2]
    converted_xy = np.stack([x, y]).T @ rot
    rx = [ix + sweep_start_position[0] for ix in converted_xy[:, 0]]
    ry = [iy + sweep_start_position[1] for iy in converted_xy[:, 1]]
    return rx, ry


def search_free_grid_index_at_edge_y(grid_map, from_upper=False):

    y_index = None
    x_indexes = []

    if from_upper:
        x_range = range(grid_map.height)[::-1]
        y_range = range(grid_map.width)[::-1]
    else:
        x_range = range(grid_map.height)
        y_range = range(grid_map.width)

    for iy in x_range:
        for ix in y_range:
            if not grid_map.check_occupied_from_xy_index(ix, iy):
                y_index = iy
                x_indexes.append(ix)
        if y_index:
            break

    return x_indexes, y_index


def setup_grid_map(ox, oy, resolution, sweep_direction, offset_grid=10):

    width = math.ceil((max(ox) - min(ox)) / resolution) + offset_grid
    height = math.ceil((max(oy) - min(oy)) / resolution) + offset_grid
    center_x = (np.max(ox) + np.min(ox)) / 2.0
    center_y = (np.max(oy) + np.min(oy)) / 2.0

    grid_map = GridMap(width, height, resolution, center_x, center_y)
    grid_map.print_grid_map_info()
    grid_map.set_value_from_polygon(ox, oy, 1.0, inside=False)
    grid_map.expand_grid()

    x_inds_goal_y = []
    goal_y = 0
    if sweep_direction == SweepSearcher.SweepDirection.UP:
        x_inds_goal_y, goal_y = search_free_grid_index_at_edge_y(
            grid_map, from_upper=True)
    elif sweep_direction == SweepSearcher.SweepDirection.DOWN:
        x_inds_goal_y, goal_y = search_free_grid_index_at_edge_y(
            grid_map, from_upper=False)

    return grid_map, x_inds_goal_y, goal_y


def sweep_path_search(sweep_searcher, grid_map, grid_search_animation=False):

    # search start grid
    c_x_index, c_y_index = sweep_searcher.search_start_grid(grid_map)
    if not grid_map.set_value_from_xy_index(c_x_index, c_y_index, 0.5):
        print("Cannot find start grid")
        return [], []

    x, y = grid_map.calc_grid_central_xy_position_from_xy_index(c_x_index,
                                                                c_y_index)
    px, py = [x], [y]

    fig, ax = None, None
    if grid_search_animation:
        fig, ax = plt.subplots()
        # for stopping simulation with the esc key.
        fig.canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

    while True:
        c_x_index, c_y_index = sweep_searcher.move_target_grid(c_x_index,
                                                               c_y_index,
                                                               grid_map)

        if sweep_searcher.is_search_done(grid_map) or (
                c_x_index is None or c_y_index is None):
            print("Done")
            break

        x, y = grid_map.calc_grid_central_xy_position_from_xy_index(
            c_x_index, c_y_index)

        px.append(x)
        py.append(y)

        grid_map.set_value_from_xy_index(c_x_index, c_y_index, 0.5)

        if grid_search_animation:
            grid_map.plot_grid_map(ax=ax)
            plt.pause(1.0)

    return px, py


def planning(ox, oy, resolution,
             moving_direction=SweepSearcher.MovingDirection.RIGHT,
             sweeping_direction=SweepSearcher.SweepDirection.UP,
             ):

    sweep_vec, sweep_start_position = find_sweep_direction_and_start_position(
        ox, oy)

    rox, roy = convert_grid_coordinate(ox, oy, sweep_vec,
                                       sweep_start_position)

    grid_map, x_inds_goal_y, goal_y = setup_grid_map(rox, roy, resolution,
                                                     sweeping_direction)

    sweep_searcher = SweepSearcher(moving_direction, sweeping_direction,
                                   x_inds_goal_y, goal_y)

    px, py = sweep_path_search(sweep_searcher, grid_map)

    rx, ry = convert_global_coordinate(px, py, sweep_vec,
                                       sweep_start_position)

    print("Path length:", len(rx))

    return rx, ry


def planning_animation(ox, oy, resolution, outer_ox, outer_oy):  # pragma: no cover
    px, py = planning(ox, oy, resolution)
    px.insert(0, ox[0])
    py.insert(0, oy[0])
    # find closest border point to ending px, py
    closest_ind = 0
    closest_to_start_ind = 0
    end_pt = np.array((px[-1], py[-1]))
    start_pt = np.array((px[0], py[0]))
    dist1 = float('inf')
    dist2 = float('inf')
    for i in range(len(ox)):
        cur_pt = np.array((ox[i], oy[i]))
        d1 = np.linalg.norm(cur_pt - end_pt)
        d2 = np.linalg.norm(cur_pt - start_pt)
        if d1 < dist1:
            dist1 = d1
            closest_ind = i
        if d2 < dist2:
            dist2 = d2
            closest_to_start_ind = i
    print("reverse path points", closest_ind, closest_to_start_ind)
    # now we have closest path, traverse drone to end
    if closest_to_start_ind < len(ox) and closest_to_start_ind > closest_ind:
        for i in range(closest_ind, closest_to_start_ind + 1):
            px.append(ox[i])
            py.append(oy[i])
    else:
        for i in range(closest_ind, len(ox)):
            px.append(ox[i])
            py.append(oy[i])
        for i in range(0, closest_to_start_ind + 1):
            px.append(ox[i])
            py.append(oy[i])

    # print(px, py)
    counter = 0
    with open('wpts.csv', mode='w') as wpts_file:
        wpts_writer = csv.writer(wpts_file, delimiter=",")

        for ipx, ipy in zip(px, py):
            if counter % 10 == 0:
                wpts_writer.writerow([ipx, ipy])

    # animation
    if do_animation:
        for ipx, ipy in zip(px, py):
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(outer_ox, outer_oy)
            # plt.plot(ox, oy, "-xb")
            plt.plot(px, py, "-r")
            plt.plot(ipx, ipy, "or")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.1)

        plt.cla()
        plt.plot(outer_ox, outer_oy)
        # plt.plot(ox, oy, "-xb")
        plt.plot(px, py, "-r")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.1)
        plt.close()


def coverage_area(ox, oy):
    coords = list(zip(ox, oy))
    lines = [[coords[i-1], coords[i]] for i in range(len(coords))]

    # Note: with 20% the polygon becomes a multi-polygon, so a loop for plotting would be needed.
    factor = 0.05

    xs = [i[0] for i in coords]
    ys = [i[1] for i in coords]
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)

    min_corner = geometry.Point(min(xs), min(ys))
    max_corner = geometry.Point(max(xs), max(ys))
    center = geometry.Point(x_center, y_center)
    shrink_distance = center.distance(min_corner)*factor

    # assert abs(shrink_distance - center.distance(max_corner)) < 0.0001

    my_polygon = geometry.Polygon(coords)
    my_polygon_shrunken = my_polygon.buffer(+shrink_distance)

    x, y = my_polygon_shrunken.exterior.xy
    l_x = list(x)
    l_y = list(y)
    for i in range(len(l_x)):
        l_x[i] = int(l_x[i])
        l_y[i] = int(l_y[i])
    return [l_x, l_y]


def main():  # pragma: no cover
    '''
    ox: x-coordinates of the corner points
    oy: y-coordinates of the corner points
    width: the width between two parallel paths
    covered_ox: x-coordinate of the corner points coverage area
    covered_oy: y-coordinate of the corner points coverage area
    '''
    print("start!!")

    # uncomment a block to see the path trajectory and other information
    # when making a custom input please ensure last and first pts are same

    # ox = [0.0, 100.0, 100.0, 40.0, -10.0, 0.0]
    # oy = [0.0, 0.0, 50.0, 80.0, 65.0, 0.0]
    # width = 4
    # p = coverage_area(ox, oy)
    # covered_ox = p[0]
    # covered_oy = p[1]
    # planning_animation(ox,oy, width, covered_ox, covered_oy)

    # ox = [0.0, 100.0, 100.0, 40.0, 0.0]
    # oy = [0.0, 0.0, 50.0, 80.0, 0.0]
    # width = 3
    # p = coverage_area(ox, oy)
    # covered_ox = p[0]
    # covered_oy = p[1]
    # planning_animation(ox,oy, width, covered_ox, covered_oy)

    ox = [0.0, 20.0, 50.0, 200.0, 120.0, 40.0, 0.0]
    oy = [0.0, -80.0, 0.0, 30.0, 60.0, 80.0, 0.0]
    width = 5.0
    p = coverage_area(ox, oy)
    covered_ox = p[0]
    covered_oy = p[1]
    planning_animation(ox, oy, width, covered_ox, covered_oy)

    if do_animation:
        plt.show()
    print("done!!")


if __name__ == '__main__':
    main()
