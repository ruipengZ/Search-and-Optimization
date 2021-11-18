import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

random_bar = False

fig = plt.figure()
ims = []
frames = []


class Node:
    def __init__(self, x, y, parent, cost):
        """
        The Node is defined by its location, parent and cost.
        :param pos: position
        :param parent: its parent
        :param cost: its cost from source to itself
        """
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost


class AStarSearch:
    def __init__(self, x_o, y_o, grid_size, step_radius):
        """
        Do A* Search on an initialized map
        :param x_o: x coordinates of obstacle
        :param y_o: y coordinates of obstacle
        :param grid_size: size of every grid
        :param step_radius: step size taking every time
        """

        self.grid_size = grid_size
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0

        self.motion = self.get_motion_mode()
        self.step_radius = step_radius

        self.min_x = round(min(x_o))
        self.min_y = round(min(y_o))
        self.max_x = round(max(x_o))
        self.max_y = round(max(y_o))

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


    @staticmethod
    def get_heuristic(node_1, node_2):
        return np.sqrt((node_1.x - node_2.x)**2+(node_1.y - node_2.y)**2)

    @staticmethod
    def get_motion_mode():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, np.sqrt(2)],
                  [-1, 1, np.sqrt(2)],
                  [1, -1, np.sqrt(2)],
                  [1, 1, np.sqrt(2)]]
        return motion

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

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def search(self, start_x, start_y, goal_x, goal_y):
        start_node = Node(self.get_xy_index(start_x, self.min_x), self.get_xy_index(start_y, self.min_y), -1, 0)
        goal_node = Node(self.get_xy_index(goal_x, self.min_x), self.get_xy_index(goal_y, self.min_y), -1, 0)
        explored = dict()
        frontier = dict()

        frontier[self.get_grid_index(start_node)] = start_node
        while True:
            if len(frontier) == 0:
                print("Frontier is empty, fail to find the path")
                break

            current_node_index = min(frontier, key=lambda idx:frontier[idx].cost + self.get_heuristic(frontier[idx], goal_node))
            current_node = frontier[current_node_index]

            # show the searched nodes
            search_x = self.get_grid_pos_1d(current_node.x, self.min_x)
            search_y = self.get_grid_pos_1d(current_node.y, self.min_y)
            frame = plt.scatter(search_x, search_y, c = "y", marker="x", s=40)
            frames.append(frame)
            ims.append(frames.copy())

            if current_node.x == goal_node.x and current_node.y == goal_node.y:
                print("Reach the goal, finish searching")
                goal_node.parent = current_node.parent
                goal_node.cost = current_node.cost
                break
            del frontier[current_node_index]
            explored[current_node_index] = current_node

            for i, _ in enumerate(self.motion):
                new_node = Node(current_node.x + self.motion[i][0],
                            current_node.y + self.motion[i][1],
                            current_node_index,
                            current_node.cost + self.motion[i][2])
                new_node_index = self.get_grid_index(new_node)

                # If the new node is out of boundary, do nothing
                if not self.verify_node(new_node):
                    continue

                if new_node_index in explored:
                    continue

                if new_node_index not in frontier:
                    frontier[new_node_index] = new_node  # discovered a new node
                else:
                    if frontier[new_node_index].cost > new_node.cost:
                        frontier[new_node_index] = new_node

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
    start_x = 10
    start_y = 10
    goal_x = 50
    goal_y = 50
    grid_size = 2
    robot_radius = 1

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
        for i in range(board_start, board_start+20):
            x_o.append(30)
            y_o.append(i)
        for i in range(board_end-40, board_end):
            x_o.append(i)
            y_o.append(30)

    plt.scatter(x_o, y_o, marker='o', c = "k", s=30)
    plt.scatter(start_x, start_y,marker=(5, 1), c = "m", s=100)
    plt.scatter(goal_x, goal_y, marker=(5, 1), c = "b", s=100)
    plt.axis("equal")
    plt.axis('off')

    A_star = AStarSearch(x_o, y_o, grid_size, robot_radius)
    x_path, y_path = A_star.search(start_x, start_y, goal_x, goal_y)

    # frame = plt.plot(x_path, y_path, "-r")
    FPS = 20
    frame_length = 1/FPS
    pause_second = 2
    for _ in range(int(pause_second / frame_length)):
        frame = plt.scatter(x_path, y_path, marker='x', c="r")
        frames.append(frame)
        ims.append(frames.copy())

    ani = animation.ArtistAnimation(fig, ims, interval=.001)
    ani.pause()
    writer = PillowWriter(fps=FPS)
    ani.save("./gif/AStar.gif", writer=writer)


if __name__ == '__main__':
    main()