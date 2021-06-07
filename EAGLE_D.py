## This is a test Library
## Graph Comression for 1st and 2nd Eigen Value

# Import Libraries
import networkx as nx
import numpy as np
from scipy import sparse as sp
from scipy import io as sc
import matplotlib.pyplot as plt

## Normalizing a MATRIX
def NormalizeMatrix(Matrix):
    row_sums = Matrix.sum(axis=1)
    return Matrix / row_sums

## Function to Extract the Tree from the DAG
# TheDepth-first search
def TreeExtraction(DAG,Tree_Option,StartingNode):
    DAG = sp.csr_matrix(DAG)                                      # Compressed Sparse Row matrix
    G=nx.DiGraph(DAG)                                             # Create The Directed Graph
    
    ## Switching between DFS and BFS
    if Tree_Option=='dfs':
          tree = nx.dfs_tree(G, StartingNode)                                # Extract the DFS Tree
    else:
          tree = nx.bfs_tree(G, StartingNode)                                # Extract the BFS Tree
         
#    tree = nx.dfs_tree(G, 0)                                     # Extract the DFS Tree
#    tree = nx.bfs_tree(G, 0)                                     # Extract the BFS Tree
    tree_matrix=nx.to_numpy_matrix(tree)                          # Graph adjacency matrix as a NumPy matrix.
    # sc.savemat('TemporaryStore.mat', {'Tree_DAG':tree_matrix})  # Delete this temporary Matrix at the end of analysis
    return tree_matrix,tree

## Function for Eigenvalue Entry
def eigenDAG(DAG,DAG_Size,Top_k_Eigenvalue_Number):
    # Matrix to Sparse
    DAG = sp.csr_matrix(DAG)
    # Create Graph
    G=nx.DiGraph(DAG)  # Create The Directed Graph
    # CAlculate Directed laclacian
    Laplacian=nx.directed_laplacian_matrix(G, nodelist=None, weight='weight', walk_type=None, alpha=0.95)
    # Normalize the matrix
    # Laplacian=NormalizeMatrix(Laplacian)
    # Eigen value of Laplacian
    eigenvalues,eigenvectors = np.linalg.eig(Laplacian)
    # Sorting the eigenvalues
    np.matrix.sort(eigenvalues)
    # Top K EigenValues
    Top_k_Eigenvalue=eigenvalues[(DAG_Size-Top_k_Eigenvalue_Number):DAG_Size]
    
    ## If the test is for 2nd Eigen Value then this line will choose the 2nd one otherwise 1st one
    Top_k_Eigenvalue=Top_k_Eigenvalue[0]         
    
    # Getting the index for Max value
    Top_k_Eigenvalue_Index = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i])[-2:]
    
    # List of Top Eigen Vactors
    Top_k_Eigenvector=np.zeros
    Top_k_Eigenvector=eigenvectors[:,Top_k_Eigenvalue_Index[0]]
    for i in range(Top_k_Eigenvalue_Number-1):
          Top_k_Eigenvector=np.column_stack((Top_k_Eigenvector,eigenvectors[:,Top_k_Eigenvalue_Index[i+1]]))
    

    return Top_k_Eigenvalue,Top_k_Eigenvector,Top_k_Eigenvalue_Index,Laplacian

## Store Eigen Values for all the Test Case
## Calculate Eigen Values for Edge Deletion
def eigenStore(DAG,DAG_Size,Top_k_Eigenvalue_Number):
    OriginalEigen,Original_Top_k_Eigenvector,Original_Top_k_Eigenvalue_Index,Original_Laplacian=eigenDAG(DAG,DAG_Size,Top_k_Eigenvalue_Number)
    EigenStoreSize=np.count_nonzero(DAG)#np.sum(DAG).astype(int)
    # Define Initials
    # Tracking the DAG Edges
    DAG_Track=np.zeros((DAG_Size,DAG_Size))
    
    # Tracking the Eigen Changes
    if Top_k_Eigenvalue_Number==1:
          EigenChange=np.zeros((Top_k_Eigenvalue_Number  , EigenStoreSize))   # 1st Eigen
    elif Top_k_Eigenvalue_Number==2:
          EigenChange=np.zeros((Top_k_Eigenvalue_Number-1, EigenStoreSize))   # 2nd Eigen
          
    # Save the DAG as a Dummy mat file
    sc.savemat('Dummy_DAG.mat', {'DAG':DAG})
    
    count=0;
    for i in range(DAG_Size):
        for j in range(DAG_Size):
            # Load DAG
            DAG=sc.loadmat('Dummy_DAG.mat')
            DAG=DAG['DAG']
            
            if DAG[i,j]>0:
                   DAG[i,j]=0
                   Top_k_Eigenvalue,Top_k_Eigenvector,Top_k_Eigenvalue_Index,Laplacian=eigenDAG(DAG,DAG_Size,Top_k_Eigenvalue_Number)
                   EigenChange[:,count]=np.absolute(OriginalEigen-Top_k_Eigenvalue)/OriginalEigen*100 #*10000
                   DAG_Track[i,j]=EigenChange[:,count]
                   count=count+1
#                   print (count)
    return EigenChange,DAG_Track,OriginalEigen

## Updated DAG from DAG Track
def NewDAG_EigenBased(DAG_Size,DAG_Track,EigenChange,OriginalEigen,CompressionPercent,DAG):
    DAG_Updated=np.zeros((DAG_Size,DAG_Size))
    for i in range(DAG_Size):
        for j in range(DAG_Size):
            if DAG_Track[i,j]>(OriginalEigen*CompressionPercent):
                DAG_Updated[i,j]=DAG[i,j]
    return DAG_Updated

## Updated DAG from DAG Track
def NewDAG_IterationBased(DAG_Size,DAG_Track,EigenChange,OriginalEigen,DAG):
    DAG_Updated=np.zeros((DAG_Size,DAG_Size))
    for i in range(DAG_Size):
        for j in range(DAG_Size):
            if DAG_Track[i,j]>np.min(EigenChange):
                DAG_Updated[i,j]=DAG[i,j]
    return DAG_Updated

## Any Edge on the DFS Tree won't be deleted
def TreeConnecting(tree_matrix,Updated_DAG,DAG_Size,DAG):
#    tree_matrix,tree=TreeExtraction(DAG)
    Updated_Tree_DAG = Updated_DAG
    for i in range(DAG_Size):
        for j in range (DAG_Size):
            if (tree_matrix[i,j]>1):
                if (Updated_DAG[i,j]==0):
                    Updated_Tree_DAG[i,j]=DAG[i,j]
    return Updated_Tree_DAG
    
    
    #Plotting the Graph
def plot_Graph(DAG, pos, Names, use = True):    
    # Add Names
    G = nx.DiGraph(DAG)  # Create default Graph
    G=nx.relabel_nodes(G,Names, copy=False) # Names of the Nodes
    if use:
        pos
    else:
        pos = nx.spectral_layout(G)
    #pos=nx.spring_layout(G)
    nx.draw(G,pos = pos,node_size=2000, width = 1.5, node_color = '#98FB98', edge_color='#8eabb7')
    nx.draw_networkx_labels(G,pos=pos, font_color = 'k', font_size = 10)
    plt.axis('off')
    return pos

# Plotting the reduction of Edges
def plot_Edge_Reduction(NumberofEdges,LabelName,mark,Color):
    ## Plotting the Number of Edges Left
    plt.plot(NumberofEdges.T,'gx-',label=LabelName,marker=mark,color=Color)
    plt.grid(True)
    plt.legend(loc=1)
    plt.title('Graph  Compression for Different DAG\'s')
    plt.ylabel('Number of Edges')
    plt.xlabel('Iteration')

## Combining Everything (Eigen Based)
def GraphCompression_EigenBased(DAG,CompressionPercent,DAG_Size,Top_k_Eigenvalue_Number,IterationNumber,tree_matrix,Tree_Connect,StartingNode):
    NumberofEdges=np.zeros((1,IterationNumber))
    for i in range(IterationNumber):
        NumberofEdges[:,i]=np.count_nonzero(DAG)#np.sum(DAG).astype(int)
#        plt.imshow(DAG)
#        plt.pause(0.5)
        EigenValue,DAG_Track,OriginalEigen=eigenStore(DAG,DAG_Size,Top_k_Eigenvalue_Number)
        DAG=NewDAG_EigenBased(DAG_Size,DAG_Track,EigenValue,OriginalEigen,CompressionPercent,DAG)
        #DAG=TreeConnecting(tree_matrix,DAG,DAG_Size)
        
        # Do we consider the tree case or not?
        if Tree_Connect=='True':
              DAG2=TreeConnecting(tree_matrix,DAG,DAG_Size,DAG)
        else:
              DAG2=DAG
        
#    plt.imshow(DAG_New)
    return DAG2,EigenValue,NumberofEdges

## Combining Everything (Eigen Based)
def GraphCompression_IterationBased(DAG,CompressionPercent,DAG_Size,Top_k_Eigenvalue_Number,IterationNumber,tree_matrix,Tree_Connect,StartingNode):
    NumberofEdges=np.zeros((1,IterationNumber))
    for i in range(IterationNumber):
        NumberofEdges[:,i]=np.count_nonzero(DAG)#np.sum(DAG).astype(int)
#        plt.imshow(DAG)
#        plt.pause(0.5)
        EigenValue,DAG_Track,OriginalEigen=eigenStore(DAG,DAG_Size,Top_k_Eigenvalue_Number)
        DAG=NewDAG_IterationBased(DAG_Size,DAG_Track,EigenValue,OriginalEigen,DAG)
        #DAG=TreeConnecting(tree_matrix,DAG,DAG_Size)
        
        # Do we consider the tree case or not?
        if Tree_Connect=='True':
              DAG2=TreeConnecting(tree_matrix,DAG,DAG_Size,DAG)
        else:
              DAG2=DAG
        
#    plt.imshow(DAG_New)
    return DAG2,EigenValue,NumberofEdges