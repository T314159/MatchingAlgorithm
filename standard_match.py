import networkx as nx

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_max_utility_graph(males, females, male_preferences, female_preferences, current_mapping):
    g = nx.Graph()
    pos = {}
    # Update position for node from each group
    for i, node in enumerate(males):
        g.add_node(node,pos=(0,len(males)-i))
    for i, node in enumerate(females):
        g.add_node(node,pos=(1,len(females)-i))

    # Make edges
    for key, value in current_mapping.items():
        g.add_edge(key,value)

    # Plot text for the males
    for i, male in enumerate(males):
        plt.text(-0.1,len(males)-i,s=male_preferences[i], horizontalalignment='right')

    # Plot text for the females
    for i, female in enumerate(females):
        plt.text(1.1,len(females)-i,s=female_preferences[i], horizontalalignment='left')

    nx.draw(g, pos=nx.get_node_attributes(g,'pos'), with_labels=True)
    plt.show()
    print('_____________________________________________________________________________')


class Male():
    def __init__(self, name, preferences):
        self.name = name
        self.preferences = preferences
        self.left = preferences
        self.matched = None


class Female():
    def __init__(self, name, preferences):
        self.name = name
        self.preferences = preferences
        self.offered = []
        self.matched = None


if __name__ == '__main__':


    males = [Male("1", [2,3,1]), Male("2", [2,3,1]), Male("3", [3,2,1])]
    females = [Female("1", [2,3,1]), Female("2", [3,1,2]), Female("3", [1,2,3])]


    change = True
    round = 1
    while change:
        print(f"Round {round}")
        change = False
        #Reset female offers
        for female in females:
            female.matched = None
            female.offered = []

        for i, male in enumerate(males):
            proposal = male.left[0]
            females[proposal-1].offered.append(i+1)
            print(f"Male {i+1} proposed to Female {proposal} ")

        for i, female in enumerate(females):
            for suitor in female.preferences[:]:
                # accepted top one,
                if suitor in female.offered:
                    if female.matched == None:
                        female.matched = suitor
                        print(f"Female {i+1} matches with male {suitor} ")
                    else:
                        males[suitor-1].left.remove(i+1)
                        print(f"Female {i+1} rejects male {suitor} ")
                        change = True

        round += 1

    #





# Does the algorithm work for weighed, non ranked - I think the answer is yes.
# Yeah definitely yes - just maybe more difference between stable and optimal.
# Is there any difference in this for regular too tho - difference in stable versus optimal

# individual choice
# collective maximization
# pairwise maximization

# cost of proposal
# nonbipartite



