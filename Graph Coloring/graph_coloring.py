#We will use genetic algorithms (GAs) to solve the graph coloring problem. The problem takes as input a graph G=(V,E) and an integer k which designates the numbers of different colors (labels) to be used. A sucessful coloring assigns each vertex v in V a color in {1,2,3,...,k} such that all adjacent vertices have different colors (i.e. for all vertices v in V if there exists an edge e in E between v_1 and v_2 then v_1 and v_2 must have different colors). The general problem of graph coloring is computationally intractable for k>2, so we will use the GA to search heuristically for solutions. 


# ## Representation
# 
# The input for each graph will be a file. The first line of each file will indicate how many nodes there are in the graph, n, and the second line will indicate how many colors to use, k, and the remainder of the file will be a list of edges represented as ordered pairs (the two connected vertices seperated by a space) with ones edge per line. 


# ### Imports 

# - `random` gives us a way to generate random bits;
# - `numpy` supports mathematical operations in multidimensional arrays;
# - `base` gives us access to the Toolbox and base Fitness;
# - `creator` allows us to create our types;
# - `tools` grants us access to the operators bank;
# - `algorithms` enables us some ready generic evolutionary loops;
# - `math` allows us to use basic mathematical functions.



import random
import numpy as np
from deap import base, creator, tools, algorithms
import math


# ### Setting up




creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# ### File I/O and Graph Data Structure
# 
# Before creating individuals, we first must read in an arbitrary graph  into a data structure. We will use adjacency list representation.  

# To support this representation we will use two classes. First, the `AdjNode` class will be used to represent the nodes of the graph. This class has three attributes, the value of the vertex, the first adjacent node in the list, and the color of the node. The second class is `Graph` and will be used to store all of the nodes and their adjacency lists. The `Graph` class has three attributes, the number of nodes n, the number of colors $k$ and the array `graph` that will store the nodes. Additionally, there is a helper function `add_edge(s,d)`, that when supplied with two nodes (`s` and `d`) will add an edge to the graph.

class AdjNode:
    def __init__(self, value):
        self.vertex = value
        self.next = None
        self.color = None


class Graph:
    def __init__(self, num, colors):
        self.n = num
        self.k = colors
        self.graph = [None] * self.n
        
    def get_node(self, s):
        return self.graph[s]
        
    # Add edges
    def add_edge(self, s, d):
        node = AdjNode(d)
        node.next = self.graph[s]
        self.graph[s] = node

        node = AdjNode(s)
        node.next = self.graph[d]
        self.graph[d] = node


# 
# function `init_graph(file)` creates a graph from a text file. 
# The input of `init_graph(file)` will be a path to a graph file and the output will be a `Graph` object. 




def init_graph(file):
    '''
    Function to read in graph file and create a graph object. 
    Inputs: 
        file: path to a graph file in /graphs
    Outputs:
        graph: Graph object containing edges given in file
    '''
    n = -1
    k = -1
    a = -1
    b = -1
    check = -1
    g = Graph(0,0)
    with open(file) as file:
        for line in file:
            for word in line.split():
                word.rstrip()
                if(check == -1 and n!=-1 and k!=-1):
                    g = Graph(n,k)
                    check = 1
                if(n  == -1):
                    n = int(word)
                elif(k == -1):
                    k = int(word)
          
                elif(a == -1):
                    a = int(word)
                elif(b == -1):
                    b = int(word)
                    g.add_edge(a,b)
                    a = -1
                    b = -1
   
    
    
    return g


# ### Creating Individuals
# 
# the first position corresponds to Node 1, the second to Node 2, etc. Then the value (allele) at each position will correspond to the color. This bit string encoding will require ceil (log_2k) bits for each gene. This scheme has the drawback that for some values of k not all bit values will correspond to a legal color. Those individuals with an invalid coloring will have a fitness of 0. 

# We will need to register the boolean attribure as well as the individual and population with the DEAP toolbox. We first instantiate the toolbox and then register the boolean attributes since they are both independent of the given graph. Next, we create the function `register_ind(graph)` which takes a graph and registers the correct size of individual to the toolbox. This function will also register the population given the indiviudal. This function will be used later in the main method, but will be helpful now in testing the fitness function. 


toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)

def register_ind(graph):
    ##get value for n and k from graph
    n = graph.n
    k = graph.k
    
    ##calculate the size of each individual
    ind_size = math.ceil(math.log2(k))*n
    
    ##register individual and population with toolbox
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# ### Fitness Function
# 
# For each individual, we want to know how close its encoded coloring is to a legal coloring of the graph ( a legal coloring is one where no nodes connected by an edge have the same color). Let G=(V,E) be a graph with n nodes and m edges. If c(i) is the assigned color of node i, then a coloring of a graph is C(i)=[c(0),c(1),...c(n)]. For a given edge i,j in E, let the function delta(i,j) = 1 if c(i) !=c(j)$, and delta(i,j)=0 if c(i)=c(j). 

# `eval_graph(graph, individual)` computes the fitness of an individual given a graph and individual. The input will be a graph object and an individual object. The output will be the total fitness of the indiviudal. 


def isvalid(n,k):
   
    if n >=0 and n < k:
        return 1
    else:
        return 0

def eval_graph(graph, individual):
    '''
    Function to compute the fitness of an individual for the graph coloring problem. 
    Inputs: 
        individual: individual object from DEAP toolbox
        graph: graph object containing nodes and edges
    Outputs:
        fitness: fitness of an individual coloring
    '''
    n = graph.n
    k  = graph.k
    size = math.ceil(math.log2(k))
    count = 0
    curr = 0
    color = []
    for i in individual:
        
        curr = curr*2 + i
        count = count + 1
        if(count == size):
            color.append(curr)
            curr = 0
            count = 0
    totcount = 0
   
    numedge = 0
    for i in range(0,n):
        currnode = graph.get_node(i)
        check1 = isvalid(color[i],k)
        c1 = color[i]
        
        while currnode != None:
            id = currnode.vertex
            c2 = color[id]
            check2 = isvalid(c2,k)
            if(check1 == 0 or check2 == 0):
                fitness = 0.0
                return fitness,
            if(check1 == 1 and check2 == 1 and c1 != c2):
                    totcount+=1 
           
            numedge = numedge + 1
            currnode = currnode.next
   
    fitness = totcount/numedge
    return fitness,
# g = Graph(4,3)
# g.add_edge(1,3)
# g.add_edge(0,2)
# g.add_edge(2,1)
# print(eval_graph(g,np.array([0,0,0,0,0,1,1,0])))




#  Update Toolbox and Evolve Population
# 
# ``cxpb`` is the probability with which two individuals are crossed, ``mutpb`` is the probability for mutating an individual, ``indpb`` is the the independent probability of each attribute to be mutated, ``tournsize`` is the size of each tournament, ``n`` is the population size, and  ``ngen`` is the number of generations. 

def main(file):
    import numpy
   
    graph = init_graph(file)
    register_ind(graph)
    
    toolbox.register("evaluate", eval_graph, graph)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.60)
    toolbox.register("select", tools.selTournament, tournsize=5)
    
    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.6, ngen=90, stats=stats, halloffame=hof, verbose=True)
    
    
    return pop, logbook, hof




if __name__ == "__main__":
    pop, log, hof = main("graphs/graph_4.txt")
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    
    
    import matplotlib.pyplot as plt
    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()
