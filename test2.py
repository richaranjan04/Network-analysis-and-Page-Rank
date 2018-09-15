import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import csv
import scipy as sp
import numpy as np
import numpy as np
from scipy.sparse import csc_matrix

G=nx.DiGraph()
with open('nodes.csv', 'r') as nodecsv: # Open the nodes file 
    nodereader = csv.reader(nodecsv) # Read the csv file
    
    nodes = [n for n in nodereader][1:]
node_names = [n[0] for n in nodes] # Get a list of the node names
with open('edges.csv', 'r') as edgecsv: # Open the edges file
    edgereader = csv.reader(edgecsv) # Read the csv
    edges = [tuple(e) for e in edgereader][1:] # Retrieve the data
G.add_nodes_from(node_names)
G.add_edges_from(edges)
print(nx.info(G))
v=nx.spring_layout(G)
plt.axis('off')
nx.draw(G,pos=v,with_labels=False,node_size = 20)
plt.show()  #Draw the graph
density = nx.density(G)
print("Network density:", density)
print("\n")
print("***---------Centrality Measures ----------------***") 
print("\n")
indeg=nx.in_degree_centrality(G)
outdeg=nx.out_degree_centrality(G)
print("\n\n In-Degree centrality: \n",indeg)
print("\n\n Out-Degree centrality: \n",outdeg)
bet=nx.betweenness_centrality(G)
print("\n\nNetwork betweenness:", bet)
close=nx.closeness_centrality(G)
print("\n\nNetwork Closeness:\n",close)
length = len(G)

print("\n\n")

print("\n\n Total number of nodes : ", length)
l = G.in_degree(node_names)
print('\n\n Indegree of all the nides are : \n',l)
m = G.out_degree(node_names)
print('\n\n Outdegree of all the nodes are : \n',m)

print("\n")
print("***---------Degree,Proximity and Rank Prestige ----------------***") 
print("\n")
x = [(name, v/(length - 1)) for (name, v) in l]
print("\n\nDegree prestige of the nodes are :\n",x)
pin=[]
for y in node_names:
    b=0
    for z in node_names:
        b=b+nx.shortest_path_length(G, source=y, target=z, weight=None)
    pin.insert(len(pin),b)
pp=[]
for q in pin:
    pp.append(q/length-1)

print("\n\nProximity prestige of the nodes are : \n",pp)

A = nx.adjacency_matrix(G)          #Adjacency Matrix

E = np.array(A.toarray())
C = np.array(A.toarray()).tolist()

def rankpres(G):
    
    n = G.shape[0]

    
    A = csc_matrix(G,dtype=np.float)
    rsums = np.array(A.sum(1))[:,0]
    ri, ci = A.nonzero()
    A.data /= rsums[ri]

   
    sink = rsums==0

   
    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r-ro)) > .0001:
        ro = r.copy()
        
        for i in range(0,n):
            
            Ai = np.array(A[:,i].todense())[:,0]
            
            Di = sink / float(n)
            
            Ei = np.ones(n) / float(n)

            r[i] = ro.dot( Ai*1 + Di*1)

    
    return r/float(sum(r))
print("\n\nRank Prestige of the nodes are :\n",rankpres(E))
print("\n\nAdjacency Matrix :\n",A.toarray())
print("\n") 
print("***---------Co-citation and Biblographic Coupling----------------***") 
print("\n") 
i = int(input("Enter node one(i): "))
j = int(input("Enter node two(j): "))

result=0
k=0
for k in range(length):
    result=result+(C[k][i]*C[k][j])
    k=k+1

print("\n\n Co-citation of the nodes :\n",result)
result1=0
k=0
for k in range(length):
    result1=result1+(C[i][k]*C[j][k])
    k=k+1

print("\n\n Biblographic Coupling of the nodes :\n",result1)

print("\n") 
print("***---------Page Rank ----------------***") 
print("\n") 

def pageRank(G, s = .85, maxerr = .0001):
    n = G.shape[0]

    
    A = csc_matrix(G,dtype=np.float)
    rsums = np.array(A.sum(1))[:,0]
    ri, ci = A.nonzero()
    A.data /= rsums[ri]

   
    sink = rsums==0

    
    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r-ro)) > maxerr:
        ro = r.copy()
        
        for i in range(0,n):
            # inlinks of state i
            Ai = np.array(A[:,i].todense())[:,0]
            
            Di = sink / float(n)
            
            Ei = np.ones(n) / float(n)

            r[i] = ro.dot( Ai*s + Di*s + Ei*(1-s) )

    # return normalized pagerank
    return r/float(sum(r))
print("\n\nPageRank of the nodes are :\n",pageRank(E))
