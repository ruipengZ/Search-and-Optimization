### Monte-Carlo Tree Search on Tree(n children) MDP
import random
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

parser = argparse.ArgumentParser()
parser.add_argument("--n_children", type=int, default=2, help="number of children of every node")
parser.add_argument("--n_depth", type=int, default=5, help="depth of tree")
parser.add_argument("--gamma", type=float, default=0.1, help="decay rate of reward")
parser.add_argument("--budget", type=int, default=30, help="max iterations of Monte-Carlo process")
parser.add_argument("--epsilon", type=float, default=0.3, help="epsilon-greedy policy")

args = parser.parse_args()
n_children = args.n_children
n_depth = args.n_depth
gamma = args.gamma
budget = args.budget
eps = args.epsilon

MIN = 0
MAX = 1
PAYOFF = 2

ims = []
frame = []
fig = plt.figure(figsize=(16, 7))

random.seed(1)


class Node():
    def __init__(self, type, depth, x, y, parent, payoff, state):
        self.type = type  # MAX or CHANCE
        self.state = state
        self.depth = depth
        self.children = []
        self.parent = parent
        self.n_visit = 0
        self.u = 0
        self.q = 0
        self.cur_child_idx = 1
        self.payoff = payoff

        ## for visualization
        self.x = x
        self.y = y
        self.plt_v_text = None

        interval = 10 * n_children ** (n_depth - self.depth)
        self.pos_list = [-(n_children - 1) * interval / 2]
        for _ in range(n_children - 1):
            self.pos_list += [self.pos_list[-1] + interval]


class MCTS():
    def __init__(self, tree_depth, c):
        self.tree_depth = tree_depth
        self.root = Node(MAX, 0, 0, 0, None, None, 0)
        self.draw_tree(self.root)
        self.payoff = [random.choice([-1,1]) for _ in range(n_children**(n_depth-1))]
        self.c = c

    def draw_tree(self, node):
        plt1 = None
        plt3 = None
        if node.type == MAX:
            color = 'g'
        elif node.type == MIN:
            color = 'b'
        else:
            color = 'grey'
            message = 'Root WON' if node.payoff == 1 else 'Root LOST'
            plt1 = plt.annotate(message, xy=(node.x, node.y), xytext=(node.x - 5, node.y - 1.5), weight='heavy')
            frame.append(plt1)

        plt2 = plt.scatter(node.x, node.y, c = color, marker='o', s=400)
        frame.append(plt2)

        if node.parent:
            plt3 = plt.arrow(node.parent.x, node.parent.y-1, node.x-node.parent.x, node.y+2-node.parent.y)
            frame.append(plt3)

        return plt1, plt2, plt3

    def expand(self, node):
        payoff = None
        child = Node(1-node.type, node.depth+1, node.x + node.pos_list[node.cur_child_idx-1], node.y - 10, node, payoff, node.state*2+node.cur_child_idx)
        node.children.append(child)
        node.cur_child_idx += 1
        self.draw_tree(child)

        ims.append(frame.copy())
        return child

    def best_child(self, s):
        max = -np.inf
        best_child = None
        for child in s.children:
            value = child.q/child.n_visit + self.c*np.sqrt(2*np.log(s.n_visit)/child.n_visit)
            if value>max:
                max = value
                best_child = child
        return best_child

    def tree_policy(self, node):
        s = node
        if s.depth+1 == self.tree_depth:
            payoff = self.payoff[s.state-n_children**(n_depth-2)-1]
            child = Node(PAYOFF, s.depth + 1, s.x, s.y - 10, s, payoff, None)
            s.children.append(child)
            self.draw_tree(child)
            ims.append(frame.copy())

        while s.depth+1!=self.tree_depth:
            if len(s.children)!=n_children: # not fully expanded
                return self.expand(s)
            else:
                s = self.best_child(s)
        return s

    def default_policy(self, node):
        s = node
        temp_list= []
        while s.depth != self.tree_depth:

            if s.depth+1 == self.tree_depth:
                payoff = self.payoff[s.state-n_children**(n_depth-1)+1]
                child = Node(PAYOFF, s.depth + 1, s.x, s.y - 10, s, payoff, None)
            else:
                payoff = None
                random_child = random.randint(1, n_children)
                child = Node(1-s.type, s.depth+1, s.x + s.pos_list[random_child-1], s.y - 10, s, payoff, s.state*2+random_child)
            s = child

            plt1, plt2, plt3 = self.draw_tree(child)
            ims.append(frame.copy())
            temp_list.append(plt1)
            temp_list.append(plt2)
            temp_list.append(plt3)

        winner = MAX if s.payoff == 1 else MIN


        for item in temp_list:
            if item:
                frame.remove(item)
        ims.append(frame.copy())

        return winner

    def backup(self, node, winner):
        s = node
        while s:
            s.n_visit += 1
            s.q += 1 if s.type==winner else 0

            if s.plt_v_text in frame:  ## if previously plotted, remove them
                frame.remove(s.plt_v_text)
            s.plt_v_text = plt.text(s.x - 5, s.y, "win:%d/visit:%d" % (s.q, s.n_visit), c='r',
                                       fontsize=12, horizontalalignment='right', clip_box=dict(ec='w'))
            frame.append(s.plt_v_text)
            ims.append(frame.copy())
            frame.remove(s.plt_v_text)
            s.plt_v_text = plt.text(s.x - 5, s.y, "win:%d/visit:%d" % (s.q, s.n_visit), c='black',
                                    fontsize=12, horizontalalignment='right', clip_box=dict(ec='w'))
            frame.append(s.plt_v_text)
            ims.append(frame.copy())

            s = s.parent

    def tree_search(self, budget):
        round = 0
        while round < budget:
            print('round', round + 1)
            plt1 = plt.text(-300, 0, "Round: %d" % (round + 1), fontsize=20,
                            bbox={'fc': 'w', 'ec': 'k'})
            frame.append(plt1)
            ### MCTS ###
            s = self.tree_policy(self.root)
            winner = self.default_policy(s)
            self.backup(s, winner)
            ############
            ims.append(frame.copy())
            round += 1

        max = 0
        best_action = None
        for action in self.root.children:
            if action.q/action.n_visit > max:
                best_action = action
                max = action.q/action.n_visit
        return best_action

    def show_best(self, best_action):
        node = self.root
        plt1 = plt.scatter(node.x, node.y, c='r', marker='o', s=400)
        frame.append(plt1)
        plt2 = plt.arrow(node.x, node.y - 1, best_action.x - node.x, best_action.y + 2 - node.y, ec='r',
                         fc='r', width=0.1)
        frame.append(plt2)
        ims.append(frame.copy())


def main():
    plt.axis('off')
    plt.tight_layout()

    mcts = MCTS(n_depth, 5)
    ims.append(frame.copy())

    best_action = mcts.tree_search(budget)
    mcts.show_best(best_action)

    ani = animation.ArtistAnimation(fig, ims, interval=1)
    writer = PillowWriter(fps=2)
    ani.save("./gif/MCTS.gif", writer=writer)



if __name__ == "__main__":
    main()