### Rapid Exploring Randon Tree
import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

random_bar = False

fig = plt.figure()
ims = []
frames = []

random.seed(0)
class Node:
    def __init__(self, x, y, parent):
        """
        The Node is defined by its location, parent and cost.
        :param pos: position
        :param parent: its parent
        """
        self.x = x
        self.y = y
        self.parent = parent


class RRT:
    def __init__(self, x_o, y_o, grid_size, start_x, start_y, goal_x, goal_y):
        """
        Do RRT Search on an initialized map
        :param x_o: x coordinates of obstacle
        :param y_o: y coordinates of obstacle
        :param grid_size: size of every grid
        """

        self.grid_size = grid_size
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.step_radius = 1

        self.start_x = start_x
        self.start_y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y

        self.min_x = round(min(x_o))
        self.min_y = round(min(y_o))
        self.max_x = round(max(x_o))
        self.max_y = round(max(y_o))

        self.goal_sample_rate = 10

        # number of grid
        self.x_width = round((self.max_x - self.min_x) / self.grid_size)
        self.y_width = round((self.max_y - self.min_y) / self.grid_size)
        self.obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
        for x_grid in range(self.x_width):
            x = self.get_grid_pos_1d(x_grid, self.min_x)
            for y_grid in range(self.y_width):
                y = self.get_grid_pos_1d(y_grid, self.min_y)
                for x_o_pos, y_o_pos in zip(x_o, y_o):
                    d = np.sqrt((x_o_pos - x)**2+(y_o_pos - y)**2)
                    if d <= self.step_radius:
                        self.obstacle_map[x_grid][y_grid] = True
                        break


    def get_grid_pos_1d(self, pos, min_pos):
        return pos * self.grid_size + min_pos

    def get_grid_index(self, node):
        index = (node.y-self.min_y)*self.x_width + node.x-self.min_x
        return index

    def get_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.grid_size)

    def verify_node(self, node):
        px = self.get_grid_pos_1d(node.x, self.min_x)
        py = self.get_grid_pos_1d(node.y, self.min_y)
        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        if self.obstacle_map[math.floor(node.x)][math.floor(node.y)] or \
                self.obstacle_map[math.ceil(node.x)][math.ceil(node.y)] or \
                self.obstacle_map[math.floor(node.x)][math.ceil(node.y)] or \
                self.obstacle_map[math.ceil(node.x)][math.floor(node.y)] :
            return False

        return True

    def generate_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            while True:
                random_x = random.randint(self.min_x, self.max_x)
                random_y = random.randint(self.min_y, self.max_y)
                rnd = Node(self.get_xy_index(random_x, self.min_x), self.get_xy_index(random_y, self.min_y), -1)
                if self.verify_node(rnd):
                    break
        else:  # goal point sampling
            rnd = Node(self.get_xy_index(self.goal_x, self.min_x), self.get_xy_index(self.goal_y, self.min_y), -1)
        return rnd

    def get_dis_angle(self, src_node, des_node):
        dis = np.sqrt((src_node.x-des_node.x)**2+(src_node.y-des_node.y)**2)
        angle = np.arctan2(des_node.y-src_node.y, des_node.x-src_node.x)
        return dis, angle

    def expand(self, src_node, des_node, length=5):
        new_node = copy.deepcopy(src_node)
        dis, angle = self.get_dis_angle(new_node, des_node)

        if length is None or length > dis:
            length = dis

        for i in range(int(length/self.step_radius)):
            if self.verify_node(new_node) is False:
                break
            new_node.x += self.step_radius * np.cos(angle)
            new_node.y += self.step_radius * np.sin(angle)


        new_node.parent = self.get_grid_index(src_node)
        if self.get_grid_index(src_node) == self.get_grid_index(new_node):
            new_node = None
        return new_node

    def get_nearest_node(self, explored, des_node):
        min = np.inf
        candidate = None
        for _,node in explored.items():
            if np.sqrt((node.x-des_node.x)**2+(node.y-des_node.y)**2)<min:
                candidate = node
                min = np.sqrt((node.x-des_node.x)**2+(node.y-des_node.y)**2)
        return candidate


    def search(self):
        start_node = Node(self.get_xy_index(self.start_x, self.min_x), self.get_xy_index(self.start_y, self.min_y), -1)
        goal_node = Node(self.get_xy_index(self.goal_x, self.min_x), self.get_xy_index(self.goal_y, self.min_y), -1)

        explored = dict()
        explored[self.get_grid_index(start_node)] = start_node
        while True:
            random_node = self.generate_random_node()
            nearest_node = self.get_nearest_node(explored, random_node)
            new_node = self.expand(nearest_node, random_node)
            if new_node == None:
                continue

            # show the searched nodes
            search_x = self.get_grid_pos_1d(new_node.x, self.min_x)
            search_y = self.get_grid_pos_1d(new_node.y, self.min_y)
            to_x = self.get_grid_pos_1d(nearest_node.x, self.min_x)
            to_y = self.get_grid_pos_1d(nearest_node.y, self.min_y)
            frame = plt.arrow(search_x, search_y, to_x - search_x, to_y - search_y,
                                              ec='g', width=0.1)

            frames.append(frame)
            ims.append(frames.copy())

            explored[self.get_grid_index(new_node)] = new_node

            if abs(new_node.x - goal_node.x)<1 and abs(new_node.y - goal_node.x)<1:
                print("Reach the goal, finish searching")
                goal_node.parent = self.get_grid_index(new_node)
                break

        x_path, y_path = self.get_final_path(goal_node, explored)
        return x_path, y_path


    def get_final_path(self, goal_node, explored):
        # generate final searched path
        x_path, y_path = [self.get_grid_pos_1d(goal_node.x, self.min_x)], [self.get_grid_pos_1d(goal_node.y, self.min_y)]
        parent_index = goal_node.parent
        while parent_index != -1:
            parent_node = explored[parent_index]
            x_path.append(self.get_grid_pos_1d(parent_node.x, self.min_x))
            y_path.append(self.get_grid_pos_1d(parent_node.y, self.min_y))
            parent_index = parent_node.parent
        return x_path, y_path


def main():
    # start and goal position
    start_x = 5
    start_y = 5
    goal_x = 50
    goal_y = 50
    grid_size = 2

    # set obstacle positions
    x_o, y_o = [], []
    board_start = 0
    board_end = 60
    for i in range(board_start, board_end):
        x_o.append(i)
        y_o.append(board_start)
    for i in range(board_start, board_end):
        x_o.append(board_end)
        y_o.append(i)
    for i in range(board_start, board_end):
        x_o.append(i)
        y_o.append(board_end)
    for i in range(board_start, board_end):
        x_o.append(board_start)
        y_o.append(i)

    if random_bar:
        x_bar_len = [20,15,5,30,25]
        y_bar_len = [10,25,15]
        for _,bar in enumerate(x_bar_len):
            x_bar = random.randint(board_start, board_end - bar)
            y_bar = random.randint(board_start, board_end)
            for i in range(0, bar):
                x_o.append(x_bar+i)
                y_o.append(y_bar)

        for _,bar in enumerate(y_bar_len):
            x_bar = random.randint(board_start, board_end)
            y_bar = random.randint(board_start, board_end-bar)
            for i in range(0, bar):
                x_o.append(x_bar)
                y_o.append(y_bar +i)

    else:
        for i in range(board_start, board_start+25):
            x_o.append(15)
            y_o.append(i)
        for i in range(board_end-25, board_end):
            x_o.append(25)
            y_o.append(i)
        for i in range(board_start, board_start+25):
            x_o.append(35)
            y_o.append(i)
        for i in range(board_end-25, board_end):
            x_o.append(45)
            y_o.append(i)

    plt.scatter(x_o, y_o, marker='o', c = "k", s=30)
    plt.scatter(start_x, start_y,marker=(5, 1), c = "m", s=100)
    plt.scatter(goal_x, goal_y, marker=(5, 1), c = "b", s=100)
    plt.axis("equal")
    plt.axis('off')

    rrt = RRT(x_o, y_o, grid_size,start_x, start_y, goal_x, goal_y)
    x_path, y_path = rrt.search()

    FPS = 20
    frame_length = 1/FPS
    pause_second = 2
    for _ in range(int(pause_second / frame_length)):
        for p in range(len(x_path)-1):
            frames.append(plt.arrow(x_path[p], y_path[p], x_path[p+1] -  x_path[p], y_path[p+1] - y_path[p],
                                              ec='r', width=0.3))
        ims.append(frames.copy())

    ani = animation.ArtistAnimation(fig, ims, interval=.001)
    ani.pause()
    writer = PillowWriter(fps=FPS)
    ani.save("./gif/RRT.gif", writer=writer)


if __name__ == '__main__':
    main()