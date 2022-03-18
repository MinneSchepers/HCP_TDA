#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Topological Data Analysis 

This script is a compilation of some topological data analysis tools.

"""

__author__ = 'Fernando Nobrega & Eduarda Centeno'
__contact__ = 'f.nobregasantos@amsterdamumc.nl or e.centeno@amsterdamumc.nl'
__date__ = '2020/05/10'   ### Date it was created
__status__ = 'Production'


####################
# Review History   #
####################


####################
# Libraries        #
####################

# Standard imports
#import os
import itertools

# Third party imports 
import numpy as np # version 1.19.1
import matplotlib.pyplot as plt # version 3.3.0
import scipy.io # version 1.4.1
from sklearn import preprocessing # version 0.22.1
import networkx as nx # version 2.4
import scipy.special

#############################
# Pre-defined settings      #
#############################
# Notice that some TDA scripts are quite heavy, if you run in a server consider using nice command
#niceValue = os.nice(10)

#You can set up the maximum clique size for your analysis.
kmax=30 # dimensions - max size for the clique algorithm

# Define Functions ------------------------------------------------------------

def normalize(matrix):
    """Matrix normalization
    
    Parameters
    ----------
    matrix: numpy matrix
        
    Returns
    -------
    X_scale_maxabs: numpy matrix
        rescaled matrix
    
    """
    # For details in this normalization, see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html
    #Scale each feature by its maximum absolute value.
    #This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity.


    # This is the data I want to scale
    X_scale=np.copy(matrix)
    # This is the one I can use for the HCP
    
    max_abs_scaler = preprocessing.MaxAbsScaler()
    X_scale_maxabs = max_abs_scaler.fit_transform(X_scale)
   
    return X_scale_maxabs #X_train_minmax


def max_cliques(N, k):
    """
    
    Parameters
    ----------
    N: number of nodes of your network
    
    k: maximum size of the cliques
       
    Returns
    -------
    mclq: total number of possible cliques with size from 0 to k
    
    OBS:
    ---
    The clique algorithm is time consuming (NP) larde and dense matrixes, this functions is an attempt to deal with it
    
    """
    
    mclq = 0
    for i in range(0,k+1):
        # Notice that we sum up to k+1, since Python does not counts the last values in the range.
        mclq += scipy.special.binom(N, i)
    
    mclq = int(mclq)
    
    return mclq


def Kmaxcliques(G, kmax=kmax):
    """
    
    Parameters
    ----------
    G: networkx graph
    
    kmax: int
        number of dimensions
    
    Returns
    -------
    C: list with all cliques of the graph G with size up to kmax
    
    """
    #Depending on the analysis, we can use a timer for the computation    
    #start_time = time.time()
    #main()
    Nodes = len(G)
    
    Cliques = nx.find_cliques(G)
    Limit = max_cliques(Nodes, kmax)
    Cl=[]
    while True:
        try:
            for i in range(0,Limit):
                clq = next(Cliques)
                if len(clq) <= kmax: # IF YOU DON'T WANNA USE KMAX JUST COMMENT THIS STEP TO MAKE IT QUICKER
                    Cl.append(clq)
        except StopIteration:
            break
    # If not provided, compute maximal cliques
    #if (C is None) : C = nx.find_cliques(G)
    
    # Sort each clique, make sure it's a tuple
    C = [list(sorted(c)) for c in Cl]
    
    return C

def Kmax_all_cliques(G,kmax=kmax):
    """"
    Enumerate all cliques to a max size
    """
    C=Kmaxcliques(G)
    Sk=set()
    for k in range(0, max(len(s) for s in C)) :
        # Get all (k+1)-cliques, i.e. k-simplices, from all max cliques mc in C
        # Notice that we are usning set(c) so that we count each clique only once
        [Sk.add(c) for mc in C for c in (itertools.combinations(mc, k+1))]
        # Check that each simplex is in increasing order
        #assert(all((list(s) == sorted(s)) for s in Sk))
        # Assign an ID to each simplex, in lexicographic order
        #S.append(dict(zip(sorted(Sk), range(0, len(Sk)))))
        #Appending the number of cliques of size k+1
        Cliques= [list(i) for i in Sk]
    return Cliques






def Euler_charac(G, kmax=kmax): 
    """
    
    Parameters
    ----------
    G: networkx graph 
    
    kmax: int
        number of dimensions
    
    Returns
    -------
    summary:
        A list with a topological summary for the graph G with Euler characteristics, tau, and number of cliques for each size
    OBS:
    ---
    This function limits the number of cliques to a maximum kmax
    
    """
    
    #start_time = time.time()
    #main()
    Nodes = len(G)
    
    Cliques = nx.find_cliques(G)
    Limit = max_cliques(Nodes, kmax)
    Cl = []
    while True:
        try:
            for i in range(0,Limit):
                clq = next(Cliques)
                if len(clq) <= kmax: # IF YOU DON'T WANNA USE KMAX JUST COMMENT THIS STEP TO MAKE IT QUICKER
                    Cl.append(clq)
        except StopIteration:
            break
    # If not provided, compute maximal cliques
    #if (C is None) : C = nx.find_cliques(G)
    
    # Sort each clique, make sure it's a tuple
    C = [tuple(sorted(c)) for c in Cl]
    
    
    summary = []
    for k in range(0, max(len(s) for s in C)) :
        # Get all (k+1)-cliques, i.e. k-simplices, from all max cliques mc in C
        # Notice that we are usning set(c) so that we count each clique only once
        Sk = set(c for mc in C for c in itertools.combinations(mc, k+1))
        # Check that each simplex is in increasing order
        #assert(all((list(s) == sorted(s)) for s in Sk))
        # Assign an ID to each simplex, in lexicographic order
        #S.append(dict(zip(sorted(Sk), range(0, len(Sk)))))
        #Appending the number of cliques of size k+1
        summary.append(len(Sk))
    tau = sum(summary) # Tau gives the total number of cliques
    kmax = len(summary) # Kmax is the maximum clique size one can find
    ec = 0 # ec is the Euler characteristics
    for i in range(0,len(summary)):
        if i%2 == 0:
                ec += summary[i]
        if i%2 == 1:
                ec += -summary[i]
        #ec+=(-1)**(k % 2)*k
    #print((k))
    summary.insert(0, kmax)
    summary.insert(0, tau)
    summary.insert(0, ec)
    for i in range(kmax, 30): # I want to include new elements after kmax with zero, to say that there are no simplicies with this size, but all the outputs will be lists with the same size
        summary.insert(kmax+3, 0) # The first guy is chi, the second is tau, the third is kmax
    
    #The output will be summary starting with EC, tau, kmax, clique_0,Clique_1,Clique_2, Clique_3, and so on...
        
    return summary


def Graph_thresh(e, i): 
    """Creating a binarized graph with a specific threshold 
    
    Parameters
    ----------
    e: int
        threshold value
        
    i: numpy matrix
        connectivity matrix
    
    Returns
    -------
    temp: networkx graph
        
    Notes
    -------
    Returns a graph that maps all elements greater than e to the zero element
    
    """
    
    #"Notice that we did not normalize the data. If you want to normalize just uncomment here"
    #ScdatanGA=np.array(normalize(Aut[i]))
    
    data = i   
    cpdata = (np.copy(np.abs(data))) # be careful to always pass copy of data, othewise will change the data as well
    cpdata[(np.copy(np.abs(data))) <= (1-e)] = 0.0
   
    thresh_graph= nx.from_numpy_matrix(cpdata[:,:])
    
    return thresh_graph


def densthr(d, i,DIAGNOSTIC=False):
    """Create graph with a specific density
    
    Parameters
    ---------   
    d: float
        density value
        
    i: numpy matrix
        connectivity matrix
        
    Returns
    -------
    finaldensity: float
        final density value 
    
    G1: networkx graph
        graph with the specified density
        
    """
    
    np.fill_diagonal(i,0)
    temp = sorted(i.ravel(), reverse=True) # Will flatten it and rank corr values.
    size = len(i)
    cutoff = np.ceil(d * (size * (size-1)))
    tre = temp[int(cutoff)]
    G0 = nx.from_numpy_matrix(i)
    G0.remove_edges_from(list(nx.selfloop_edges(G0)))
    G1 = nx.from_numpy_matrix(i)
    for u,v,a in G0.edges(data=True):
        if (a.get('weight')) <= tre:
            G1.remove_edge(u, v)
    finaldensity = nx.density(G1)
    if DIAGNOSTIC==True:
        print(finaldensity)
    
    return G1


def Eulerange_thr(i, maxvalue):
    """
    computes the Euler Characteristic and the respective summary metrics for a range of thresholds
    
    Parameters
    ---------    
    i: numpy matrix
        connectivity matrix
    
    maxvalue: int
    
    Returns
    -------
    Ec: List with Euler characteristic for a list of thresholds
        
    Notes
    -------
    Filtration process based on thresh
    Notice that we sliced the network in 1/100 steps. 
    
    """

    Ec = []
    for j in range(0, maxvalue):
        Ec.append(Euler_charac(Graph_thresh(j/100, i)))
        
    return Ec


def Eulerange_dens(i, maxvalue):
    """
    Computes the Euler Characteristic and the respective summary metrics for a range of densities

    
    Parameters
    ---------    
    i: numpy matrix
        connectivity matrix
    
    maxvalue: int
    
    Returns
    -------
    Ec: List with the Euler characteristic for a list of densities
        
    Notes
    -------
    Filtration process based on density
    Notice that we sliced the network in 1/100 steps. 

    
    """

    Ec = []
    for j in range(0, maxvalue):
        G = densthr(j/1000, i)
        Ec.append(Euler_charac(G))
        
    return Ec


def Eulerchoice_thr(i, maxvalue, k):
    """
    
    Parameters
    ---------    
    i: numpy matrix
        connectivity matrix
    
    maxvalue: int
    
    k: int
        euler characteristic=0,total=1,max=2,3=vertices,4=edges,5=triangles, etc.
    
    Returns
    -------
    output: Returns a list with  an specific summary metric k for a range of thresholds.
        
    Notes
    -------
    

    """

    temp = Eulerange_thr(i, maxvalue)
    output = [temp[i][k] for i in range(0, maxvalue)]
    
    return output


def Eulerchoice_dens(i, maxvalue, k):
    """
    
    Parameters
    ---------    
    i: numpy matrix
        connectivity matrix
    
    maxvalue: int
    
    k: int
        euler characteristic=0,total=1,max=2,3=vertices,4=edges,5=triangles, etc.
    
    Returns
    -------
    output: Returns a list with  an specific summary metric k for a range of densities.
        
    Notes
    -------
    

    """

    temp = Eulerange_dens(i, maxvalue)
    output = [temp[i][k] for i in range(0, maxvalue)]
    
    return output


def plotEuler_thr(i, maxvalue):
    """Plotting the Euler entropy, i.e. the logarithm of the Euler characteristics for a given threshold interval
    
    Parameters
    ---------    
    i: numpy matrix
        connectivity matrix
    
    maxvalue: int
        from 0 to 100
    
    Returns
    -------
    A plot of the Euler entropy based on thr
        
    """
    
    plt.plot(np.log(np.abs(Eulerchoice_thr(i, maxvalue, 0)))) # Change to eulerchoice_dens if intended
    plt.xlabel('Threshold (ε)')
    plt.ylabel('Euler entropy Sχ = ln |χ(ε)|')
    locs, labels = plt.xticks()
    plt.xticks(locs, list(locs/100))
    plt.xlim(0, maxvalue)
    plt.show()
    
    
def plotEuler_den(i, maxvalue):
    """Plotting the Euler entropy, i.e. the logarithm of the Euler characteristics for a given density interval
    
    Parameters
    ---------    
    i: numpy matrix
        connectivity matrix
    
    maxvalue: int
        from 0 to 100
    
    Returns
    -------
    A plot of the Euler entropy based on density
        
    """
    
    plt.plot(np.log(np.abs(Eulerchoice_dens(i, maxvalue, 0)))) # Change to eulerchoice_dens if intended
    plt.xlabel('Threshold (ε)')
    plt.ylabel('Euler entropy Sχ = ln |χ(d)|')
    locs, labels = plt.xticks()
    plt.xticks(locs, list(locs/100))
    plt.xlim(0, maxvalue)
    plt.show()

    

def Curv_density(d, i, verbose=False):
    """Compute nodal curvature (Knill's curvature) based on density
    
    Parameters
    ---------   
    d: float
        density value
        
    i: numpy matrix
        connectivity matrix
        
    Returns
    -------
    fden: float
        final density value for the graph
    
    curv: numpy array
        array with curvature values
        
    """
    
    def DIAGNOSTIC(*params) :
        if verbose : print(*params)
    DIAGNOSTIC('This function run over all nodes and computes the curvature of the nodes in the graph' )
    
    # This is the initial Graph
    #fden, 
    G = densthr(d,i) 
    # Enumerating all cliques of G up to a certain size
    temp = Kmax_all_cliques(G)
    
    # This lista is a vector V where each v_i is the number of cliques of size i
    lista = []
    
    # We suppose that the size of the cliques are smaller than 50, so we create an empty list of size 50 for the lista
    for i in G.nodes():
        # We start with empty scores for the curvature
        lista.append([0] * 50) # creating a list of lists for each node - all empty for the scores for each size for each node
    
    DIAGNOSTIC('These are all cliques of the Network:')
    # THIS WILL PRINT ALL THE CLIQUES
    DIAGNOSTIC(temp)
    
    DIAGNOSTIC('We now print the curvature/clique score of each node in the network')
    
    # Now we run over all nodes checking if the node belongs to one clique or another
    Sc=[]
    for node in G.nodes(): # now we run the script for each clique
        score = 0 # This is the initial score of the node in the participation rank
        for clique in temp:
            # Checking the size of the clique
            k = len(clique)
            # If this node is in the clique, we update the curvature
            if node in clique:
                score+=1 # If the node is in the clique raises the score
                lista[node][k-1] += (-1)**(k+1)*1/k # Increases the curvature score for a size k with a different weight due to Gauss-Bonnet theorem - is k-1 since len>0 and python starts from zero.
        Sc.append(score)
        
        DIAGNOSTIC('The node '+str(node)+' has score ='+str(score))
    
    total=[]
    for elements in lista:
        # Summing the participation in all sizes, so that we can compute the curvature (TOTAL IS ACTUALLY THE CURVATURE - WITHOUT NORMALIZATION)
        total.append(sum(elements)) # This is good if one wants to normalize by the maximum
    DIAGNOSTIC(total)
    
    nor=sum(total) ####!!! not being used //REMOVE ?
    nor2=max(total) ####!!! not being used //REMOVE ?
    # nt is normalized by the sum
    #nt2 is normalized by the max"
    nt=[]
    nt2=[]
    
    # I just removed where one could find division by zero
    #for i in range(0,len(total)):
    #    nt.append(total[i]/nor)
    #    nt2.append(total[i]/nor2)
    most=np.argsort(-np.array(total))#
    
    #def showrank():
    for i in most:
            DIAGNOSTIC('the node ' + str(i)+ ' is in '+ str(total[i])+ ' cliques')
    #    return 
    #DIAGNOSTIC(showrank())
    
    DIAGNOSTIC('These are the most important nodes ranked according to the total clique score')
    DIAGNOSTIC(most)
    DIAGNOSTIC('These is the array nt')

    DIAGNOSTIC(nt)
    DIAGNOSTIC('These is the array nt2')

    DIAGNOSTIC(nt2)
    DIAGNOSTIC('These is the array lista')

    DIAGNOSTIC(lista)
    DIAGNOSTIC('The output is one vector normalizing the value from the maximum')
    #vector=10000*np.array(nt)
    # nor2 is the maximum- The output nt2 is in percentage - That means the max get 100 and the rest bet 0-100
    
    #curv gives the curvature  - put Sc instead of curv to get that the particiaption rank - notice that you can normalize in many ways"
    curv=[]
    for i in range(0, len(lista)):
        curv.append(sum(lista[i]))# Summing up for a fixed node all the curvature scores gives the curvature of the nodes
    curv = np.array(curv)
    # Now, the curvature is not normalized!!!
    return curv#fden, curv


def Curv_thr(e, i, verbose=False):
    """Compute nodal curvature based on threshold
    
    Parameters
    ---------   
    e: float
        threshold value
        
    i: numpy matrix
        connectivity matrix
        
    Returns
    -------
    curv: numpy array
        array with curvature values
        
    """
    
    def DIAGNOSTIC(*params):
        if verbose : print(*params)
    DIAGNOSTIC('This function run over all nodes and computes the curvature of the nodes in the graph' )
    
    # This is the initial Graph
    G = Graph_thresh(e,i)  
    temp = Kmax_all_cliques(G)
    
    # This lista is a vector V where each v_i is the number of cliques of size i
    lista = []
    
    # We suppose that the size of the cliques are smaller than 20, so we create an empty list of size 20 for the lista
    for i in G.nodes():
        lista.append([0] * 50) # creating a list of lists for each node - all empty for the scores for each size for each node
    
    DIAGNOSTIC('These are all cliques of the Network:')
    DIAGNOSTIC(temp)
    DIAGNOSTIC('We now print the curvature/clique score of each node in the network')
    
    # Now we run over all nodes checking if the node belongs to one clique or another
    Sc=[]
    for node in G.nodes(): # now we process for each clique
        score = 0 # This is the initial score of the node in the participation rank
        for clique in temp:
            k = len(clique)
            if node in clique:
                score+=1 # If the node is in the clique raises the score
                lista[node][k-1] += (-1)**(k+1)*1/k # Increases the curvature score for a size k with a different weight due to Gauss-Bonnet theorem
        Sc.append(score)
        
        DIAGNOSTIC('The node '+str(node)+' has score ='+str(score))
    
    total=[]
    for elements in lista:
        total.append(sum(elements)) # This is good if one wants to normalize by the maximum
    DIAGNOSTIC(total)
    
    nor=sum(total) ####!!! not being used //REMOVE ?
    nor2=max(total) ####!!! not being used //REMOVE ?
    # nt is normalized by the sum
    #nt2 is normalized by the max"
    nt=[]
    nt2=[]
    
    # I just removed where one could find division by zero
    #for i in range(0,len(total)):
    #    nt.append(total[i]/nor)
    #    nt2.append(total[i]/nor2)
    most=np.argsort(-np.array(total))#
    
    #def showrank():
    for i in most:
            DIAGNOSTIC('the node ' + str(i)+ ' is in '+ str(total[i])+ ' cliques')
    #    return 
    #DIAGNOSTIC(showrank())
    
    DIAGNOSTIC('These are the most important nodes ranked according to the total clique score')
    DIAGNOSTIC(most)
    DIAGNOSTIC('These is the array nt')

    DIAGNOSTIC(nt)
    DIAGNOSTIC('These is the array nt2')

    DIAGNOSTIC(nt2)
    DIAGNOSTIC('These is the array lista')

    DIAGNOSTIC(lista)
    DIAGNOSTIC('The output is one vector normalizing the value from the maximum')
    #vector=10000*np.array(nt)
    # nor2 is the maximum- The output nt2 is in percentage - That means the max get 100 and the rest bet 0-100
    
    #curv gives the curvature  - put Sc instead of curv to get that the particiaption rank - notice that you can normalize in many ways"
    curv=[]
    for i in range(0, len(lista)):
        curv.append(sum(lista[i]))# Summing up for a fixed node all the curvature scores gives the curvature of the nodes
    curv = np.array(curv)
    
    return curv


def SaveEuler(individual, name, tresh):
    """Save Euler results
    
    Parameters
    ---------           
    individual: numpy matrix
        connectivity matrix
        
    name: str
        file name
        
    tresh: float
        threshold value
        
    Returns
    -------
    Files with results
        
    """
    
    values =(Eulerchoice_thr(individual,tresh,0)) # change to eulerchoice_dens if intended
    with open(name, 'w') as output:
        output.write(str(values))
        
        
        
def Participation_in_cliques(d,i,cl,verbose=False):
    """
    Returns a list with the participation rank in cliques of a fixed size
    inputs:
    d: density
    i: matrix
    cl: clique size
    """
    def DIAGNOSTIC(*params) :
        if verbose : print(*params)
        return

    # I want that the output is a vector analogous with the previous one, but for a fixed k, not for all k
    # COMPUTING THE CLIQUES
    G = densthr(d,i)  
    temp = Kmax_all_cliques(G) 
    DIAGNOSTIC('These are all cliques')
    DIAGNOSTIC(temp)
    "This lista is a vector V where each v_i is the number of cliques of size i"
    lista=[]
    "We suppose that the size of the cliques are smaller than 50, so we create an empty list of size 50 for the lista"
    for i in G.nodes():
        lista.append([0] * 50) 
    # creating a list of lists - all empty for the Scores of the nodes
    #DIAGNOSTIC(lista)#print(list)# here I can change - Creating a list
    #test=cliques(e,i)
    #score=0
    "Now we run over all nodes checking if the node is in one clique or another"
    for node in G.nodes(): # now I have to do the process for is in clique
        score=0 # This is the score of the node
        # RUNNING FOR ALL NODES IN G
        for clique in temp:
            #RUNNING FOR ALL CLIQUES ENUMERATED
            k=len(clique)
            # CHECKING IF THERE IS A NODE WITH THIS SIZE
            if node in clique:
                #INCLUDING THE SCORE FOR THE CLIQUE
                score+=1
                lista[node][k-1]+=+1
       # print('the node '+str(node)+' has score ='+str(score))
    total=[]
    for elements in lista:
        total.append(sum(elements))
    DIAGNOSTIC('This is the number of cliques each node is participating in')
    DIAGNOSTIC(total)
    DIAGNOSTIC(np.sum(total))
    nor=sum(total)
    nt=[]
    for i in range(0,len(total)):
        nt.append(total[i]/nor)
    #vector=10000*np.array(nt)
    klist=[]
    DIAGNOSTIC('Now lets plot the number of k-cliques each number is participating in')
    for i in G.nodes():
        klist.append(lista[i][cl-1])
        DIAGNOSTIC('the node '+str(i)+ ' has '+ str(cl) + ' - score =' + str(lista[i][cl-1]))
    
    mostk=np.argsort(-np.array(klist))
    
    
    #nor=sum(total)
    #nor2=max(total)
    #nt=[]
    #nt2=[]
    #for i in range(0,len(total)):
    #    nt.append(total[i]/nor)
    #    nt2.append(total[i]/nor2)
    #most=np.argsort(-np.array(total))#
    DIAGNOSTIC('These are the most important nodes ranked according to the k-clique score')

    DIAGNOSTIC(mostk)
    #def showrank():
    #DIAGNOSTIC(mostk)
    for i in mostk:
            DIAGNOSTIC('the node ' +str(i)+ ' is in '+ str(klist[i])+ ' ' +str(cl)+ '-cliques')
    #    return 
    #DIAGNOSTIC(showrank())
    
    #DIAGNOSTIC(nt)
    #DIAGNOSTIC(nt2)
    #DIAGNOSTIC('The output is one vector normalizing the value from the maximum')
    DIAGNOSTIC(klist)
    
    
    #lista[i]=node i vector
    #print(temp)
    maxk=max(klist)
    totk=sum(klist)
    #np.nan_to_num(100*np.array(klist)/maxk)
    # We can do some choices: Here I choose the percentage of all cliques to plot
    return np.nan_to_num(100*np.array(klist)/totk)