### Tabular Q-learning on Tree(n children) MDP
import random
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

parser = argparse.ArgumentParser()
parser.add_argument("--n_children", type=int, default= 2, help="number of children of every node")
parser.add_argument("--n_depth", type=int, default= 4, help="depth of tree")
parser.add_argument("--gamma", type=float, default= 0.1, help="decay rate of reward")
parser.add_argument("--budget", type=int, default= 100, help="max iterations of Monte-Carlo process")
parser.add_argument("--epsilon", type=float, default= 0.3, help="epsilon-greedy policy")


args = parser.parse_args()
n_children = args.n_children
n_depth = args.n_depth
gamma = args.gamma
budget = args.budget
eps = args.epsilon

CHANCE = 0
STATE = 1
TERMINAL = 2

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
            self.action = None
            ## for visualization

        if self.type == CHANCE:
            self.visited = 0
            self.next_state = None
            self.Q_value = 0
            self.plt_Q_value_text = None

            


class T_QL():
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
        interval = 10 * n_children ** (self.tree_depth - node.depth)
        pos_list = [-(n_children - 1) * interval / 2]
        for _ in range(n_children - 1):
            pos_list += [pos_list[-1] + interval]

        if node.depth == self.tree_depth:
            self.state_list.append(node)
            ## add a terminal child
            node.children.append(Node(CHANCE, node.depth + 1, None, None, None, None))
            node.action = node.children[0]
            return

        if node.type == STATE:
            self.state_list.append(node)
            for i in range(n_children):
                node.children.append(Node(CHANCE, node.depth + 1, None, None, node.x + pos_list[i], node.y - 10))

        else:
            ### Generate transition probabilities
            prob_list = [random.random() for _ in range(3)]
            s = sum(prob_list)
            prob_list = [i / s for i in prob_list]
            for i in range(n_children):
                node.children.append(
                    Node(STATE, node.depth + 1, random.randint(1, 9), prob_list[i], node.x + pos_list[i], node.y - 10))

        for child in node.children:
            self.build_tree(child)


    def draw_tree(self, node):
        if node.type == STATE:
            ## plot node
            color = 'g'
            plt.scatter(node.x, node.y, c=color, marker='o', s=400)
            ## plot reward
            plt.text(node.x + 3, node.y-0.5, "%d" % node.reward, fontsize=15)

            if node.depth != self.tree_depth:
                for child in node.children:
                    ## plot arrow
                    plt.arrow(node.x, node.y - 1, child.x - node.x, child.y + 2 - node.y)
                    self.draw_tree(child)

        elif node.type == CHANCE:
            ## plot node
            color = 'b'
            plt.scatter(node.x, node.y, c=color, marker='o', s=400)
            ## plot Q value
            node.plt_Q_value_text = plt.text(node.x - 3, node.y - 0.5, "%.2f" % node.Q_value, c='r', fontsize=15,
                                             horizontalalignment='right', clip_box=dict(ec='w'))
            frame.append(node.plt_Q_value_text)

            for child in node.children:
                ## plot arrow and policy arrow
                plt.arrow(node.x, node.y - 1, child.x - node.x, child.y + 2 - node.y, ec='b')
                ## plot transition prob as ?
                plt.text((node.x + child.x) / 2, (node.y + child.y) / 2, "?", fontsize=15, c='b')
                self.draw_tree(child)

        return

    def simulate_one_step(self, node):
        ## simulate one step for state node
        # if node.depth == self.tree_depth:
        #     return
        if len(node.action.children) == 0:
            node.action.next_state = None

        elif node.type == STATE:
            ## Environment gives the next state based on probabilities
            next_state_idx = random.choices(range(n_children), [child.prob for child in node.action.children])[0]
            node.action.next_state = node.action.children[next_state_idx]
            
    def pick_action(self, node):
        if random.random() < eps:
            node.action = node.children[random.randint(0,len(node.children)-1)]
        else:
            mx=0
            for child in node.children:
                if child.Q_value >= mx:
                    mx = child.Q_value
                    node.action = child


    def Q_value_update(self, node, gamma):

        if node.depth == self.tree_depth:
            plt1 = plt.scatter(node.x, node.y, c='r', marker='o', s=400)
            frame.append(plt1)
            ims.append(frame.copy())

            ## only one action
            alpha = 10 / (9 + node.action.visited)
            node.action.Q_value = node.action.Q_value +  alpha * (node.reward - node.action.Q_value)
            node.action.visited += 1
            self.clear_frame_but_text()
            return

        else:
            self.pick_action(node)
            self.simulate_one_step(node)
            if node.action.next_state==None:
                node.action.Q_value = 0

                ## plot node
                plt1 = plt.scatter(node.x, node.y, c='r', marker='o', s=400)
                ## plot policy arrow
                plt2 = plt.arrow(node.x, node.y - 1, node.action.x - node.x,
                                 node.action.y + 2 - node.y, ec='r', width=0.1)
                ## plot chance node
                plt3 = plt.scatter(node.action.x, node.action.y, c='r', marker='o', s=400)

                frame.append(plt1)
                frame.append(plt2)
                frame.append(plt2)
                frame.append(plt3)

                ims.append(frame.copy())
                self.clear_frame_but_text()
                return

            node.action.visited += 1
            alpha = 10 / (9 + node.action.visited)
            
            mx = 0
            for child in node.action.next_state.children:
                mx = max(mx, child.Q_value)

            node.action.Q_value = node.action.Q_value + alpha * (node.reward + gamma * mx - node.action.Q_value)

            ## plot node
            plt1 = plt.scatter(node.x, node.y, c='r', marker='o', s=400)
            ## plot policy arrow
            plt2 = plt.arrow(node.x, node.y - 1, node.action.x - node.x,
                      node.action.y + 2 - node.y, ec='r', width=0.1)
            ## plot chance node
            plt3 = plt.scatter(node.action.x, node.action.y, c='r', marker='o', s=400)
            ## plot chance arrow
            plt4 = plt.arrow(node.action.x, node.action.y - 1, node.action.next_state.x - node.action.x,
                                           node.action.next_state.y + 2 - node.action.y,
                                           ec='r', width=0.1)
            ## plot next state
            plt5 = plt.scatter(node.action.next_state.x, node.action.next_state.y, c='r', marker='o', s=400)
            frame.append(plt1)
            frame.append(plt2)
            frame.append(plt2)
            frame.append(plt3)
            frame.append(plt4)
            frame.append(plt5)

        ## plot value
        if node.action.plt_Q_value_text in frame:  ## if previously plotted, remove them
            frame.remove(node.action.plt_Q_value_text)
        node.action.plt_Q_value_text = plt.text(node.action.x - 3, node.action.y-0.5, "%.2f" % node.action.Q_value, c='r', fontsize=15, horizontalalignment='right', clip_box=dict(ec='w'))
        frame.append(node.action.plt_Q_value_text)

        ## record this frame
        ims.append(frame.copy())
        self.clear_frame_but_text()

        self.Q_value_update(node.action.next_state, gamma)



    def Tabular_QL(self, gamma, budget):
        round = 0
        while round<budget:
            i=0
            while i<len(frame):
                if not isinstance(frame[i], matplotlib.text.Text):
                    frame.remove(frame[i])
                else:
                    i += 1

            print('round', round+1)
            plt1 = plt.text(-50, 0, "Round: %d" % (round+1), fontsize=20,
                            bbox={'fc': 'w', 'ec': 'k'})

            frame.append(plt1)
            state = self.state_list[random.randint(0,len(self.state_list)-1)]
            self.Q_value_update(state, gamma)
            round += 1
        return

    def show_best(self):
        for state in self.state_list:
            if state.depth!=self.tree_depth:
                mx = -1
                for child in state.children:
                    if child.Q_value > mx:
                        best_action = child
                        mx = child.Q_value
                plt1 = plt.arrow(state.x, state.y - 1, best_action.x - state.x, best_action.y + 2 - state.y, ec='r',
                                fc='r', width=0.1)
                frame.append(plt1)
        ims.append(frame.copy())


def main():
    plt.axis('off')
    plt.tight_layout()

    tab_q_l = T_QL(n_depth)
    ims.append(frame.copy())

    tab_q_l.Tabular_QL(gamma, budget)
    tab_q_l.show_best()
    ani = animation.ArtistAnimation(fig, ims, interval=1)
    writer = PillowWriter(fps=2)
    ani.save("./gif/T_QL.gif", writer=writer)


if __name__ == "__main__":
    main()