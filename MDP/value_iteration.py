### Value Iteration on Tree(n children) MDP
import random
import numpy as np
import argparse

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

parser = argparse.ArgumentParser()
parser.add_argument("--n_children", type=int, default= 2, help="number of children of every node")
parser.add_argument("--n_depth", type=int, default= 4, help="depth of tree")
parser.add_argument("--gamma", type=float, default= 0.1, help="decay rate of reward")
parser.add_argument("--epsilon", type=float, default= 0.01, help="threshold of convergence")
args = parser.parse_args()
n_children = args.n_children
n_depth = args.n_depth
gamma = args.gamma
epsilon = args.epsilon

CHANCE = 0
STATE = 1


ims = []
frames = []
fig = plt.figure(figsize=(16, 7))

random.seed(1)

class Node():
    def __init__(self, type, depth, reward, prob, x, y):
        self.type = type  # STATE or CHANCE
        self.depth = depth
        self.children = []
        self.reward = reward
        self.prob = prob
        self.x = x
        self.y = y

        if self.type == STATE:
            self.value = 0
            self.plt_value_text = None
            self.policy = None
            self.plt_policy_arrow = None



class ValueIter():
    def __init__(self, tree_depth):
        self.tree_depth = tree_depth
        self.root = Node(STATE, 0, random.randint(1, 10), None, 0, 0)
        self.delta = 0

        self.build_tree(self.root)
        self.draw_tree(self.root)

    def build_tree(self, node):
        interval = 10 * n_children ** (self.tree_depth-node.depth)
        pos_list = [-(n_children-1)*interval/2]
        for _ in range(n_children-1):
            pos_list += [pos_list[-1]+interval]

        if node.depth == self.tree_depth:
            return

        if node.type == STATE:
            for i in range(n_children):
                node.children.append(Node(CHANCE, node.depth + 1, None, None, node.x + pos_list[i], node.y-10))

        else:
            ### Generate transition probabilities
            prob_list = [random.random() for _ in range(3)]
            s = sum(prob_list)
            prob_list = [i / s for i in prob_list]
            for i in range(n_children):
                node.children.append(Node(STATE, node.depth + 1, random.randint(1, 9), prob_list[i], node.x + pos_list[i], node.y - 10))

        for child in node.children:
            self.build_tree(child)


    def draw_tree(self, node):
        if node.type == STATE:
            ## plot node
            color = 'g'
            plt.scatter(node.x, node.y, c=color, marker='o', s=400)
            ## plot reward
            plt.text(node.x + 5, node.y, "%d" % node.reward, fontsize=10)
            for child in node.children:
                ## plot arrow
                plt.arrow(node.x, node.y - 1, child.x - node.x, child.y + 2 - node.y)
                self.draw_tree(child)

        elif node.type == CHANCE:
            ## plot node
            color = 'b'
            plt.scatter(node.x, node.y, c=color, marker='o', s=400)
            for child in node.children:
                ## plot arrow
                plt.arrow(node.x, node.y - 1, child.x - node.x, child.y + 2 - node.y)
                ## plot transition prob
                plt.text((node.x + child.x) / 2 -4, (node.y + child.y) / 2, "%.2f" % child.prob, fontsize=10)
                self.draw_tree(child)

        return


    def Bellman_update(self, node, gamma):
        if node.type == STATE: ## Bellman update on state
            prev_value = node.value
            mx = 0
            best_action = None
            for action in node.children:
                if sum([child.prob*child.value for child in action.children]) >= mx:
                    mx = sum([child.prob*child.value for child in action.children])
                    best_action = action

            node.value = node.reward + gamma * mx
            print('delta', self.delta)
            if np.abs(node.value - prev_value) > self.delta:
                self.delta = np.abs(node.value - prev_value)

            ## plot value
            plt1 = plt.scatter(node.x, node.y, c='r', marker='o', s=400)
            frames.append(plt1)

            if node.plt_value_text == None:
                node.plt_value_text = plt.text(node.x - 10.5, node.y, "%.2f" % node.value, c='r', fontsize=10)
            else:
                frames.remove(node.plt_value_text)
                node.plt_value_text = plt.text(node.x - 10.5, node.y, "%.2f" % node.value, c='r', fontsize=10)

            frames.append(node.plt_value_text)
            ims.append(frames.copy())

            plt3 = plt.scatter(node.x, node.y, c='g', marker='o', s=400)
            frames.append(plt3)
            ims.append(frames.copy())

            ## plot best action
            if node.depth != self.tree_depth:
                if node.plt_policy_arrow == None:
                    node.plt_policy_arrow = plt.arrow(node.x, node.y - 1, best_action.x - node.x, best_action.y + 2 - node.y, ec='r', width=0.1)

                else:
                    if best_action != node.policy:
                        frames.remove(node.plt_policy_arrow)
                        node.plt_policy_arrow = plt.arrow(node.x, node.y - 1, best_action.x - node.x,
                                                          best_action.y + 2 - node.y, ec='r', width=0.1)

                frames.append(node.plt_policy_arrow)
                ims.append(frames.copy())

        ## recursion
        for child in node.children:
            self.Bellman_update(child, gamma)


    def value_iteration(self, gamma, epsilon):
        round = 0
        while True:
            print('round',round)
            plt1 = plt.text(-(n_children**n_depth-1)*10 / 2, 0, "Round: %d"%round, fontsize=20, bbox={'fc':'w', 'ec':'k'})
            frames.append(plt1)
            ims.append(frames.copy())
            self.delta = 0
            self.Bellman_update(self.root, gamma)
            if self.delta < epsilon * (1-gamma)/gamma:
                break
            round += 1

        print(self.root.value)
        return



def main():
    plt.axis('off')
    plt.tight_layout()

    VI = ValueIter(n_depth)
    VI.value_iteration(gamma, epsilon)

    ani = animation.ArtistAnimation(fig, ims, interval=1)
    writer = PillowWriter(fps=3)
    ani.save("./gif/value_iter.gif", writer=writer)
    plt.show()

if __name__ == "__main__":
    main()