### Minimax Search on n children Tree
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

MIN = 0
MAX = 1
TERMINAL = 2

n_children = 3
n_depth = 3

ims = []
frames = []
fig = plt.figure(figsize=(16, 7))

class Node():
    def __init__(self, type, depth, payoff, x, y):
        self.type = type # MIN or MAX or TERMINAL
        self.depth = depth
        self.children = []
        self.payoff = payoff
        self.x = x
        self.y = y


class Minimax_search():
    def __init__(self, tree_depth, root_type):
        self.tree_depth = tree_depth
        self.root = Node(root_type, 0, None, 0, 0)

        self.build_tree(self.root)
        self.draw_tree(self.root)

    def build_tree(self, node):
        interval = 10 * n_children ** (self.tree_depth - node.depth)
        pos_list = [-(n_children - 1) * interval / 2]
        for _ in range(n_children - 1):
            pos_list += [pos_list[-1] + interval]

        if node.type == TERMINAL:
            return

        elif node.depth+1 == self.tree_depth:
            for i in range(n_children):
                node.children.append(Node(TERMINAL, node.depth + 1, random.randint(1,10), node.x + pos_list[i], node.y-10))

        else:
            for i in range(n_children):
                node.children.append(Node(1-node.type, node.depth + 1, None, node.x + pos_list[i], node.y-10))

        for child in node.children:
            self.build_tree(child)


    def draw_tree(self, node):
        if node.type == MAX:
            color = 'g'
        elif node.type == MIN:
            color = 'r'
        else:
            color = 'grey'
            plt.annotate("%d" % node.payoff, xy=(node.x, node.y), xytext=(node.x - 5, node.y - 1.5), weight='heavy')

        plt.scatter(node.x, node.y, c = color, marker='o', s=400)

        for child in node.children:
            plt.arrow(node.x, node.y-1, child.x - node.x, child.y+2 - node.y)
            self.draw_tree(child)
        return

    def minimax(self, node):
        if node.type == TERMINAL:
            return node.payoff

        elif node.type == MAX:
            value = -np.inf
            for child in node.children:
                if self.minimax(child) > value:
                    value = self.minimax(child)
                    best_child = child

            plt1 = plt.arrow(node.x, node.y-1, best_child.x-node.x, best_child.y+2-node.y, ec='r', fc='r', width=0.1)
            plt2 = plt.text(node.x + 15, node.y, "%d" % value, fontsize=15)
            frames.append(plt1)
            frames.append(plt2)
            ims.append(frames.copy())
            return value

        elif node.type == MIN:
            value = np.inf
            for child in node.children:
                if self.minimax(child) < value:
                    value = self.minimax(child)
                    best_child = child

            plt1=plt.arrow(node.x, node.y-1, best_child.x-node.x, best_child.y+2-node.y, ec='r', fc='r', width=0.1)
            plt2=plt.text(node.x+15, node.y, "%d" % value, fontsize=15)
            frames.append(plt1)
            frames.append(plt2)
            ims.append(frames.copy())
            return value


def main():
    plt.axis('off')
    plt.tight_layout()

    mm_search = Minimax_search(n_depth,MAX)
    mm_search.minimax(mm_search.root)

    ani = animation.ArtistAnimation(fig, ims, interval=1)
    writer = PillowWriter(fps=3)
    ani.save("./gif/minimax.gif", writer=writer)


if __name__ == "__main__":
    main()