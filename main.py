import networkx as nx

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def match_next_male(male_index, males, females, male_preferences, female_preferences):
    for female in male_preferences[male_index]:
        if(female not in current_mapping.values()):
            current_mapping[males[male_index]] = female
            return current_mapping
        elif(female in current_mapping.values()):
            current_mapping_inverse = dict(zip(current_mapping.values(),current_mapping.keys()))
            current_male_for_the_female = current_mapping_inverse[female]
            if(female_preferences[males.index(current_male_for_the_female)] > female_preferences[male_index]):
                current_mapping[males[male_index]] = female
                current_mapping.pop(current_male_for_the_female)
                return current_mapping

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


if __name__ == '__main__':

    males = ['m1', 'm2', 'm3']
    females = ['w1', 'w2', 'w3']

    male_preferences = [['w1', 'w2', 'w3'], ['w2', 'w3', 'w1'], ['w2', 'w3', 'w1']]
    female_preferences = [['m2', 'm3', 'm1'], ['m3', 'm1', 'm2'], ['m1', 'm2', 'm3']]

    current_mapping = {}
    while(len(current_mapping) != len(males)):
        for male_index in range(len(males)):
            if(current_mapping.get(males[male_index]) is None):
                current_mapping = match_next_male(male_index, males, females, male_preferences, female_preferences)
                plot_max_utility_graph(males, females, male_preferences, female_preferences, current_mapping)

