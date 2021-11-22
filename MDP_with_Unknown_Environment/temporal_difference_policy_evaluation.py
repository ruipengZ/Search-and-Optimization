### Monte-Carlo Policy Evaluation on Tree(n children) MDP
import random
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

parser = argparse.ArgumentParser()
parser.add_argument("--n_children", type=int, default= 2, help="number of children of every node")
parser.add_argument("--n_depth", type=int, default= 6, help="depth of tree")
parser.add_argument("--gamma", type=float, default= 0.1, help="decay rate of reward")
parser.add_argument("--budget", type=int, default= 50, help="max iterations of Monte-Carlo process")

args = parser.parse_args()
n_children = args.n_children
n_depth = args.n_depth
gamma = args.gamma
budget = args.budget

CHANCE = 0
STATE = 1

ims = []
frame = []
fig = plt.figure(figsize=(16, 7))

random.seed(1)

class Node():
    def __init__(self, type, depth, reward, prob, x, y):
        self.type = type  # MAX or CHANCE
        self.depth = depth
        self.children = []

        ## for visualization
        self.x = x
        self.y = y

        if self.type == STATE:
            self.reward = reward
            self.prob = prob

            self.value = 0
            self.visited = 0
            self.policy = None
            ## for visualization
            self.plt_value_text = None

        if self.type == CHANCE:
            self.next_state = None


class TDPE():
    def __init__(self, tree_depth):
        self.tree_depth = tree_depth
        self.root = Node(STATE, 0, random.randint(1, 10), None, 0, 0)
        self.state_list = []

        self.build_tree(self.root)
        self.draw_tree(self.root)

    @staticmethod
    def clear_frame_but_text():
        i = 0
        while i < len(frame):
            if not isinstance(frame[i], matplotlib.text.Text):
                frame.remove(frame[i])
            else:
                i += 1

    def build_tree(self, node):

        if node.depth == self.tree_depth:
            self.state_list.append(node)
            return

        if node.type == STATE:
            self.state_list.append(node)
            node.children.append(Node(CHANCE, node.depth + 1, None, None, node.x, node.y - 10))
            node.policy = node.children[0]

        else:
            interval = 10 * n_children ** (self.tree_depth - node.depth//2)
            pos_list = [-(n_children - 1) * interval / 2]
            for _ in range(n_children - 1):
                pos_list += [pos_list[-1] + interval]

            ## Generate transition probabilities, unknown for the state
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
            plt.text(node.x + 12, node.y-1, "%d" % node.reward, fontsize=15)
            ## plot value
            node.plt_value_text = plt.text(node.x - 10, node.y-1, "%.2f" % node.value, c='r', fontsize=15, horizontalalignment='right', clip_box=dict(ec='w'))
            frame.append(node.plt_value_text)

            for child in node.children:
                ## plot arrow
                plt.arrow(node.x, node.y - 1, child.x - node.x, child.y + 2 - node.y)
                self.draw_tree(child)

            ## plot the fixed policy
            if node.depth != self.tree_depth:
                plt.arrow(node.x, node.y - 1, node.policy.x - node.x,
                          node.policy.y + 2 - node.y, ec='y', width=0.1)

        elif node.type == CHANCE:
            ## plot node
            color = 'b'
            plt.scatter(node.x, node.y, c=color, marker='o', s=400)
            for child in node.children:
                ## plot arrow and policy arrow
                plt.arrow(node.x, node.y - 1, child.x - node.x, child.y + 2 - node.y, ec='b')
                ## plot transition prob as ?
                plt.text((node.x + child.x) / 2, (node.y + child.y) / 2, "?", fontsize=15, c='b')
                self.draw_tree(child)

        return

    def simulate_one_step(self, node):
        ## simulate one step for state node
        if node.depth == self.tree_depth:
            return

        if node.type == STATE:
            ## Environment gives the next state based on probabilities
            next_state_idx = random.choices(range(n_children), [child.prob for child in node.policy.children])[0]
            node.policy.next_state = node.policy.children[next_state_idx]

    def value_update(self, node, gamma):

        node.visited += 1
        alpha = 10 / (9 + node.visited)
        if node.depth == self.tree_depth:
            node.value = node.value + alpha * (node.reward - node.value)

            plt1 = plt.scatter(node.x, node.y, c='r', marker='o', s=400)
            frame.append(plt1)
        else:
            self.simulate_one_step(node)
            node.value = node.value + alpha * (node.reward + gamma * node.policy.next_state.value - node.value)
            ## plot node
            plt1 = plt.scatter(node.x, node.y, c='r', marker='o', s=400)
            ## plot policy arrow
            plt2 = plt.arrow(node.x, node.y - 1, node.policy.x - node.x,
                      node.policy.y + 2 - node.y, ec='r', width=0.1)
            ## plot chance node
            plt3 = plt.scatter(node.policy.x, node.policy.y, c='r', marker='o', s=400)
            ## plot chance arrow
            plt4 = plt.arrow(node.policy.x, node.policy.y - 1, node.policy.next_state.x - node.policy.x,
                                           node.policy.next_state.y + 2 - node.policy.y,
                                           ec='r', width=0.1)
            ## plot next state
            plt5 = plt.scatter(node.policy.next_state.x, node.policy.next_state.y, c='r', marker='o', s=400)
            frame.append(plt1)
            frame.append(plt2)
            frame.append(plt2)
            frame.append(plt3)
            frame.append(plt4)
            frame.append(plt5)

        ## plot value
        if node.plt_value_text in frame:  ## if previously plotted, remove them
            frame.remove(node.plt_value_text)
        node.plt_value_text = plt.text(node.x - 10, node.y-1, "%.2f" % node.value, c='r', fontsize=15, horizontalalignment='right', clip_box=dict(ec='w'))
        frame.append(node.plt_value_text)

        ## record this frame
        ims.append(frame.copy())
        self.clear_frame_but_text()


    def Temporal_Diff_PE(self, gamma, budget):
        round = 0
        while round<budget:
            i=0
            while i<len(frame):
                if not isinstance(frame[i], matplotlib.text.Text):
                    frame.remove(frame[i])
                else:
                    i += 1

            print('round', round+1)
            plt1 = plt.text(-500, 0, "Round: %d" % (round+1), fontsize=20,
                            bbox={'fc': 'w', 'ec': 'k'})

            frame.append(plt1)
            state = self.state_list[random.randint(0,len(self.state_list)-1)]
            self.value_update(state, gamma)
            round += 1
        return

def main():
    plt.axis('off')
    plt.tight_layout()

    td = TDPE(n_depth)
    ims.append(frame.copy())
    td.Temporal_Diff_PE(gamma, budget)

    ani = animation.ArtistAnimation(fig, ims, interval=1)
    writer = PillowWriter(fps=2)
    ani.save("./gif/TD_PE.gif", writer=writer)


if __name__ == "__main__":
    main()