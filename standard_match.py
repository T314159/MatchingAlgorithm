import networkx as nx

import numpy as np
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment
import pandas as pd
import matplotlib.pyplot as plt

debug = 0


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

#nx.set_node_attributes(G, bb, "betweenness")

class Male:
    def __init__(self, name, preferences):
        self.name = name
        self.preferences = preferences
        if debug==1: print(preferences)
        self.left = preferences[:]
        self.matched = None


class Female:
    def __init__(self, name, preferences):
        self.name = name
        self.preferences = preferences
        if debug==1: print(preferences)
        self.offered = []
        self.matched = None


class Matching:
    def __init__(self, males = [], females = []):
        self.males = males
        self.females = females

    def generate_preferences(self, n):
        from random import sample
        self.males = [Male(i, sample(list(range(1, n+1)), n)) for i in range(1, n+1)]
        self.females = [Female(i, sample(list(range(1, n+1)), n)) for i in range(1, n+1)]

    def make_graph(self):
        self.graph = nx.Graph()
        for i, male in enumerate(self.males):
            self.graph.add_node("m" + str(i+1), pos=(0, len(self.males) - i))
        for i, female in enumerate(self.females):
            self.graph.add_node("w" + str(i+1), pos=(1, len(self.females) - i))

    def find_all_matchings(self):
        self.stable_matchings = []
        overall_outcomes = []

        import itertools
        for matching in list(itertools.permutations(list(range(0,len(self.males))))):
            for i in range(len(self.males)):
                self.males[i].matched   = matching[i]+1
                self.females[matching[i]].matched = i+1

            good = True
            i=0
            while i < len(self.males) and good:
                for female_index in self.males[i].preferences:
                    if self.males[i].matched == female_index: break
                    else:
                        if self.females[female_index-1].preferences.index(i+1) < self.females[female_index-1].preferences.index(self.females[female_index-1].matched):
                            good = False

                i+=1

            if good:
                self.stable_matchings.append(matching)
                metric = self.metrics()
                overall_outcomes.append((metric["males_avg"]+metric["females_avg"])/2)

        if debug == 1: print("All matchings: ", self.stable_matchings)
            # need to add metrics in here too
        return ({"best_outcome": min(overall_outcomes), "num_outcome": len(overall_outcomes),
                 "overall_avg": sum(overall_outcomes) / len(overall_outcomes)
                 })

    def finding_matching_foundamental(self):
        change = True
        round = 1
        while change:
            if debug==1: print(f"Round {round}")
            change = False
            # Reset female offers
            for female in self.females:
                female.matched = None
                female.offered = []

            for i, male in enumerate(self.males):
                proposal = male.left[0]
                self.females[proposal - 1].offered.append(i + 1)
                if debug == 1: print(f"Male {i + 1} proposed to Female {proposal} ")

            for i, female in enumerate(self.females):
                for suitor in female.preferences[:]:
                    # accepted top one,
                    if suitor in female.offered:
                        if female.matched == None:
                            female.matched = suitor
                            if debug == 1: print(f"Female {i + 1} matches with male {suitor} ")
                        else:
                            self.males[suitor - 1].left.remove(i + 1)
                            if debug == 1: print(f"Female {i + 1} rejects male {suitor} ")
                            change = True

            round += 1
        for i, female in enumerate(self.females):
            if debug == 1: print("Adding edge from " + "m" + str(female.matched) + " to " + "w" + str(i+1))
            self.graph.add_edge("m" + str(female.matched), "w" + str(i+1))

        return round

    def finding_matching_hungarian(self):

        # Convert to the right matrix form
        matrix_list = []
        for i, male in enumerate(self.males):
            row = []
            for j, female in enumerate(self.females):
                male_pref = male.preferences.index(j+1)
                female_pref = female.preferences.index(i+1)
                row.append(2*male_pref+female_pref)
            matrix_list.append(row)

        matrix = np.array(matrix_list)
        if debug == 1: print(matrix)

        # Run hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(matrix)
        if debug == 1: print(row_ind, col_ind)
        # Adding matches
        for male, female in enumerate(col_ind):
            self.males[i].matched = female+1
            self.females[female].matched = male+1

        # Add edges to our graph structure
        for i, female in enumerate(self.females):
            if debug == 1: print("Adding edge from " + "m" + str(female.matched) + " to " + "w" + str(i+1))
            self.graph.add_edge("m" + str(female.matched), "w" + str(i+1))


    def display_matching(self):
        nx.draw(self.graph, pos=nx.get_node_attributes(self.graph, 'pos'), with_labels=True)
        plt.show()

    def metrics(self, method="linear"):
        if method == "linear":
            females_outcomes = []
            males_outcomes = []

            for i, female in enumerate(self.females):
                females_outcomes.append(female.preferences.index(female.matched))
                males_outcomes.append(self.males[female.matched-1].preferences.index(i+1))
            return({"males_outcomes" : males_outcomes, "females_outcomes" : females_outcomes,
                    "males_avg": sum(males_outcomes)/len(males_outcomes),
                    "females_avg": sum(females_outcomes)/len(females_outcomes),
                    })





if __name__ == '__main__':

    # males = [Male("1", [2,3,1]), Male("2", [2,3,1]), Male("3", [3,2,1])]
    # females = [Female("1", [2,3,1]), Female("2", [3,1,2]), Female("3", [1,2,3])]
    test = "hungarian"

    if test == "foundamental":
        ns = []
        male_avgs = []
        female_avgs = []
        for n in range(2, 50):
            iterations = 1000

            males_outcomes = []
            females_outcomes = []

            for i in range(iterations):
                print(i)
                matching = Matching()
                matching.generate_preferences(n)
                matching.make_graph()
                matching.finding_matching_foundamental()
                metrics = matching.metrics()
                males_outcomes.append(metrics["males_avg"])
                females_outcomes.append( metrics["females_avg"])

            print(f"\n\nn={n}")
            # print("  Males avg:", round(np.mean(males_outcomes),3), "Std: ", round(np.std(males_outcomes),3))
            # print("Females avg:", round(np.mean(females_outcomes),3), "Std: ", round(np.std(females_outcomes),3))

            ns.append(n)
            male_avgs.append(np.mean(males_outcomes))
            female_avgs.append(np.mean(females_outcomes))
            #print("\n", stats.ttest_ind(males_outcomes, females_outcomes, equal_var = False))

    elif test == "hungarian":
        ns = []
        male_avgs = []
        female_avgs = []
        for n in range(250, 251, 25):
            iterations = 500

            males_outcomes = []
            females_outcomes = []

            for i in range(iterations):
                matching = Matching()
                matching.generate_preferences(n)
                matching.make_graph()
                matching.finding_matching_hungarian()
                metrics = matching.metrics()
                males_outcomes.append(metrics["males_avg"])
                females_outcomes.append( metrics["females_avg"])

            print(f"\n\nn={n}")
            print("  Males avg:", round(np.mean(males_outcomes),3), "Std: ", round(np.std(males_outcomes),3))
            print("Females avg:", round(np.mean(females_outcomes),3), "Std: ", round(np.std(females_outcomes),3))

            ns.append(n)
            male_avgs.append(np.mean(males_outcomes))
            female_avgs.append(np.mean(females_outcomes))
            print("\n", stats.ttest_ind(males_outcomes, females_outcomes, equal_var = False))

        print(ns)
        print("Average male matching outcome:", male_avgs)
        print("Average female matching outcome:", female_avgs)

    elif test == "brute_force":
        ns = []
        overall_avgs = []
        best_overalls = []
        for n in range(9, 10):
            print(f"\n\nn={n}")
            iterations = 500

            overall_avg_outcomes = []
            best_overall_outcomes = []


            for i in range(iterations):
                if i % 10 == 0: print(i)
                matching = Matching()
                matching.generate_preferences(n)
                matching.make_graph()
                metrics = matching.find_all_matchings()
                overall_avg_outcomes.append(metrics["overall_avg"])
                best_overall_outcomes.append(metrics["best_outcome"])

            # print("  Males avg:", round(np.mean(males_outcomes),3), "Std: ", round(np.std(males_outcomes),3))
            # print("Females avg:", round(np.mean(females_outcomes),3), "Std: ", round(np.std(females_outcomes),3))

            ns.append(n)
            overall_avgs.append(np.mean(overall_avg_outcomes))
            best_overalls.append(np.mean(best_overall_outcomes))
            # print("\n", stats.ttest_ind(males_outcomes, females_outcomes, equal_var = False))

        print(ns)
        print("Average stable matching outcome:", overall_avgs)
        print("Best stable matching:", best_overalls)



# Does the algorithm work for weighed, non ranked - I think the answer is yes.
# Yeah definitely yes - just maybe more difference between stable and optimal.
# Is there any difference in this for regular too tho - difference in stable versus optimal

# individual choice
# collective maximization
# pairwise maximization

# cost of proposal
# nonbipartite



