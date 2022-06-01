import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from graphviz import render, Source

'''
C = [1,1,2] #Configuration Vector C^0
F = np.array([[1,1,0],[0.5,0.5,0],[0,0,1],[0,0,0.5]]) #Production Function Matrix
FL = np.array([[1,0],[1,0],[0,1],[0,1]]) # Function  Location Matrix
syn = [(0,1),(1,0)]


no_func = np.array([0,0,0,0])

T =  np.array([0,0,0,4])
has_threshold = [0,0,0,1]
'''
'''
C = [1,2,0,0,
    5,0,0,0,0] #Configuration Vector C^0    
F = np.array([[1,0,0,0,0,0,0,0,0],[-1,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],
              [0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,0,1,0]
                ]) #Production Function Matrix
FL = np.array([[1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],
              [0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0]
            ]) # Function  Location Matrix
syn = [(0,2),(1,3),(3,6),(2,6),(4,5),(5,6),(6,7)]

no_func = np.array([0,0,0,0,0,0,0,0,0,0])

T =  np.array([0,0,0,0,0,0,0,0,-1,1,0])
has_threshold = [0,1,0,1,0,0,0,0,1,1,0]
'''
'''
C = [1,1,1,0,0,2]#Configuration Vector C^0
F = np.array([[0.5,0.5,0,0,0,0],[0,0,1,0,0,0],[0,0,-1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,-1,0],[0,0,0,0,0,0.5]]) #Production Function Matrix
FL = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]) # Function  Location Matrix
T =  np.array([1,0,0,0,0,0]) # Threshold Vector
no_func = np.array([0,0,0,0,0,0,0,0,0])
syn = [(0,1),(1,0),(0,2),(1,3),(2,4),(3,4)] # synapse matrix
'''


#Addition Module
'''
C = [1,-1,0,0,0,0,0,0,0]#Configuration Vector C^0
F = np.array([[1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,-1,0,0,0,0,0],[0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,0.5,0,0,0],[0,0,0,0,0,0,-0.5,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]]) #Production Function Matrix
FL = np.array([[1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]]) # Function  Location Matrix
T =  np.array([0,0,0,0,0,0,0,0,0]) # Threshold Vector
no_func = np.array([0,1,0,0,0,0,0,0,0])
syn = [(0,2),(0,3),(0,4),(2,5),(3,5),(3,6),(4,1),(4,6),(5,7),(6,8)] # synapse matrix
'''
'''
C = [100,50,0,0]
F = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
FL = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
T =  np.array([0,0,1]) 
no_func = np.array([0,0,0,1])
syn = [(0,2),(1,2),(2,3)]


#Subtraction Module
C = [1,1,0,0,0,0,0,0,0,0,0,0,0]#Configuration Vector C^0
F = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,-1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0.5,0,0,0,0,0,0,0],[0,0,0,0,0,0,-1,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,2,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,1]]) #Production Function Matrix
FL = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,1]]) 
T =  np.array([0,1,0,0,0,2,0,0,0,0,0,0,0]) 
no_func = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
syn = [(0,1),(0,2),(1,5),(1,6),(2,3),(2,5),(3,4),(3,7),(4,5),(5,8),(5,9),(5,10),(6,5),(7,11),(8,11),(9,12),(10,5)]
'''

'''
#Fin Module
C = [1,-2,0,0,0,0,0,0]
F = np.array([[1,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,-1]])
FL = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
T =  np.array([0,1,0,1,0,0,1,0])
no_func = np.array([0,0,0,0,0,0,0,0])
syn = [(0,2),(0,3),(0,4),(1,6),(1,3),(1,7),(2,5),(3,6),(4,1),(4,3),(5,7),(6,3),(6,1)]
'''
'''
C = [1,0,0,0,0,0,0,0]#Configuration Vector C^0
F = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0]])
'''
#C is 2n +  4 elements

S = [9,8,7,6]
SUM = 13
conf=[]
for i in range(2*len(S)+6):
    if (i<len(S)):
        conf.append(S[i])
    elif (i>len(S) and i <=2*len(S)):
        conf.append(0)
    elif i ==2*len(S)+1:
        conf.append(SUM+1)
    elif i == 2*len(S) + 2:
        conf.append(0)
    elif i == 2*len(S) + 3:
        conf.append(0)
    elif i == 2*len(S) + 4:
        conf.append(0)
    elif i == 2*len(S) + 5:
        conf.append(0)

print (conf)

C = [1,2,3,0,0,0,
     7,0,0,0,
     0]
#C = conf

#first len(S) array alloted for nondeterministic choice
F = np.array([[1,0,0,0,0,0,0,0,0,0,0],[-1,0,0,0,0,0,0,0,0,0,0], 
              [0,1,0,0,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0,0,0,0],[0,0,-1,0,0,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,0,0,0,1,0]
              ])
FL = np.array([[1,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0], 
               [0,1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0],
               [0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0],
               [0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,1,0]
               ])

no_func = np.array([0,0,0,0,0,0,0,0,0,0,0])

T =  np.array([0,0,0,0,0,0,0,0,0,0,0,-1,1,0])
has_threshold = [0,0,0,0,0,0,1,1,1,0,0,1,1,0]

syn = [(0,3),(1,4),(2,5),(3,8),(4,8),(5,8),(6,7),(7,8),(8,9),(9,10)]




num_neuron = FL.shape[1]
num_func = FL.shape[0]
num_var = F.shape[1]
print(num_neuron,num_func,num_var)
#returns list of variables for a given function
def getVars(index_function):
    vars = []
    for i in range(0,F.shape[1]):
        if F[index_function][i] != 0:
            vars.append(i)
    return vars

#check all variables in index_function if greater than threshold i

def checkThreshold(c, index_function):
    vars = getVars(index_function)
    #print(vars)
    if has_threshold[index_function]:

        for var in vars:
            if c[var] >= T[index_function]:
                continue
            else:
                return False
    else:
        return True
    return True

#get functions present in neuron m
def getFunctions(m,Active):
    functions = []
    for i in range(0,num_func):
        if Active[i][m]: 
            functions.append(i)
    return functions
#generateSpiking matrix and create functionswasUsed matrix
def generateSpikingMatrix(configuration):
    Active = FL.copy()
    num_possible = []
    for i in range(0,num_neuron):
        count = 0
        for j in range(0,num_func):
            if FL[j][i] == 1:
                if checkThreshold(configuration,j):
                    count += 1
                    Active[j][i] = 1
                    functionWasUsed.append(1)

                else:
                    Active[j][i] = 0
                    functionWasUsed.append(0)
            else:
                Active[j][i] = 0
                #functionWasUsed.append(0)
        num_possible.append(count)


    q = 1

    for i in num_possible:
        if i != 0:
            q = q*i

    S = np.zeros((q,num_func), dtype = int)

    q_i = q
    for m in range(0,num_neuron):
        function = getFunctions(m,Active)
        if num_possible[m] == 0:
            for j in function:
                for k in range (0,q):
                    S[k][j] = 0
            continue
        else:
            i = 0
            p = q_i/num_possible[m]
            while i < q:
                for j in function:
                    k = 0
                    while k < p:
                        S[i][j] = 1
                        k += 1
                        i += 1
        q_i = q_i /num_possible[m]
    return(S)

#get neuron index given an input function index
def getNeuronFromFunction(index_function):
    for j in range(0,num_neuron):
        if FL[index_function][j]:
            return j
#get neuron index given an input variable
def getNeuronFromVar(var):
    for i in range(0,num_func):
        if F[i][var] != 0:
            return getNeuronFromFunction(i)
#generates the production matrix
def generateProductionMatrix(configuration):
    PM = np.zeros((num_func,num_var), dtype = int)
    for i in range(0,num_func):
        sum = 0
        for j in range(0,num_var):
            sum = sum + F[i][j]*configuration[j]

        m = getNeuronFromFunction(i)

        for var in range(0,num_var):
            if (no_func[var]):
                k = var
            else:
                k = getNeuronFromVar(var)
            if (m,k) in syn:
                PM[i][var] = sum
    return PM
#returns a list of used variables from an input of used functions
def UsedVariables(usedFunctionList):
    usedVars = []
    for i in range(0,num_var):
        usedVars.append(0)

    for i in range(0,num_func):
        #print(num_func)
        if usedFunctionList[i] == 1:
            vars = getVars(i)
            for var in vars:
                usedVars[var] = 1
    for i in range(0,len(no_func)):
        if no_func[i] == 1:
            usedVars[i] = 0


    return usedVars

UnexploredStates = [C]
ExploredStates = []
depth = 2
curr_depth = 0

#list of Node objects representing various configurations
historyNode = []
historyNode.append(Node(str(C)))

while (UnexploredStates != []):
    nextStates = []
    nextRemove = []

    for configuration in UnexploredStates:

        #converts a possible numpy list to normal python list
        if isinstance(configuration,list):
            pass
        else:
            configuration = configuration.tolist()

        #generate spiking and production matrix
        functionWasUsed = []
        S = generateSpikingMatrix(configuration)
        vars_used = UsedVariables(functionWasUsed)
        PM = generateProductionMatrix(configuration)


        net_gain = np.matmul(S,PM)
        q_next = net_gain.shape[0]
        C_old =  np.zeros((q_next,num_var), dtype = int)

        #if variable is unused the value of the variable will be maintained
        for i in range(0,q_next):
            for j in range(0,num_var):
                if vars_used[j] == 0:
                    C_old[i][j] = configuration[j]
        #print(C_old,vars_used)
        #print(PM)
        C_next = np.add(C_old,net_gain)
        C_next = np.unique(C_next,axis =0)

        #set rows in C_next to be children of configuration
        if ExploredStates == []:
            min_node_index = 0
            max_node_index = 1
        
        for i in range(min_node_index,max_node_index+1):
            if historyNode[i].name == str(configuration):
                parent = i
                break

        for row in C_next:
            #print(row.tolist())
            if ExploredStates == []:
                nextStates.append(row.tolist())
                node = Node(str(row.tolist()),parent = historyNode[parent])
                historyNode.append(node)
                continue
            else:
                node = Node(str(row.tolist()),parent = historyNode[parent])
                historyNode.append(node)
                nextStates.append(row.tolist())
        
        #print("parent is:"+str(i))
        #print("children is/are:"+str(historyNode[i].children))
        ExploredStates.append(configuration)
        nextRemove.append(configuration)

        #print("explored states: "+str(ExploredStates))
    max_node_index = len(historyNode)-1
    min_node_index = max_node_index-len(nextStates)

    for state in nextRemove:
        UnexploredStates.remove(state)

    #print(nextStates,type(nextStates))

    for state in nextStates:
        #if not already in ExploredStates append
        if state  not in ExploredStates:
            UnexploredStates.append(state)
    if (UnexploredStates == []):
        break
    if curr_depth < depth: 
        curr_depth += 1
    else:
        break

for pre, fill, node in RenderTree(historyNode[0]):
    print("%s%s" % (pre, node.name))

#export tree object to Dot format for visualization
DotExporter(historyNode[0]).to_dotfile("test.dot")