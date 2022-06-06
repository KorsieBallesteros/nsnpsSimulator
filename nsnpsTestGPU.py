import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from graphviz import render, Source
import sys
import gen_ss_det
from gen_ss_non_det import genNonDetConfiguration, genNonDetSubsetSumFunctionLocationMatrix, genNonDetSubsetSumFunctionMatrix, genNonDetSubsetSumNoFunc, genNonDetSubsetThresholdMatrix, genNonDetSubsetSynapseList
from gen_ss_det import genConfiguration, genDetSubsetSumFunctionLocationMatrix, genDetSubsetSumFunctionMatrix, genDetSubsetSumNoFunc, genDetSubsetThresholdMatrix, genDetSubsetSynapseList
import time
import json
from numba import cuda, types, float32, int32
from math import ceil


#choose between non_determinisic solution and deterministic

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
                #k = getNeuronFromVar(var)
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
        if usedFunctionList[i] == 1:
            vars = getVars(i)
            for var in vars:
                usedVars[var] = 1
    for i in range(0,len(no_func)):
        if no_func[i] == 1:
            usedVars[i] = 0


    return usedVars

def naiveMatrixMult(A,B):
    result = [[sum(a * b for a, b in zip(A_row, B_col))
                        for B_col in zip(*B)]
                                for A_row in A]
    return result
TPB = 32
#sum = 0
@cuda.jit
def naive_matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp
        
@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=int32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=int32)

    x, y = cuda.grid(2)
    #sum = sum + 1
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid
    #print(x,y,tx,ty)

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()
        
        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

results = []
with open(r'test_cases.txt', 'r') as fp:
    test_cases = json.load(fp)

time_sum_gen = 0 
for instance in test_cases:
    print(instance)
    set = instance[0]
    target_sum = instance[1]
    input_size = len(set)
    
    gen_start = time.time()
    '''
    C = genNonDetConfiguration(set,target_sum)
    FL = np.array(genNonDetSubsetSumFunctionLocationMatrix(input_size))
    F = np.array(genNonDetSubsetSumFunctionMatrix(input_size))
    no_func = genNonDetSubsetSumNoFunc(input_size)
    T,has_threshold = genNonDetSubsetThresholdMatrix(input_size)
    syn = genNonDetSubsetSynapseList(input_size)
    '''
    '''
    C = genConfiguration(set,target_sum)
    FL = np.array(genDetSubsetSumFunctionLocationMatrix(input_size))
    F = np.array(genDetSubsetSumFunctionMatrix(input_size))
    no_func = genDetSubsetSumNoFunc(input_size)
    T,has_threshold = genDetSubsetThresholdMatrix(input_size)
    syn = genDetSubsetSynapseList(input_size)
    '''
    
    if sys.argv[1] == 'gen_ss_non_det':

        C = genNonDetConfiguration(set,target_sum)
        FL = np.array(genNonDetSubsetSumFunctionLocationMatrix(input_size))
        F = np.array(genNonDetSubsetSumFunctionMatrix(input_size))
        no_func = genNonDetSubsetSumNoFunc(input_size)
        T,has_threshold = genNonDetSubsetThresholdMatrix(input_size)
        syn = genNonDetSubsetSynapseList(input_size)
    
    elif sys.argv[1] == 'gen_ss_det':
        C = genConfiguration(set,target_sum)
        FL = np.array(genDetSubsetSumFunctionLocationMatrix(input_size))
        F = np.array(genDetSubsetSumFunctionMatrix(input_size))
        no_func = genDetSubsetSumNoFunc(input_size)
        T,has_threshold = genDetSubsetThresholdMatrix(input_size)
        syn = genDetSubsetSynapseList(input_size)
    
    gen_end = time.time()
    time_sum_gen += (gen_end-gen_start)
  
    num_neuron = FL.shape[1]
    num_func = FL.shape[0]
    num_var = F.shape[1]



                
    UnexploredStates = [C]
    ExploredStates = []
    depth = 10
    curr_depth = 0

    #list of Node objects representing various configurations
    historyNode = []
    historyNode.append(Node(str(C)))

    time_sum_spik = 0
    time_sum_ng = 0
    time_sum_pm = 0
    
    program_start = time.time()
    
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
            
            #spiking matrix computation
            start = time.time()
            S = generateSpikingMatrix(configuration)
            end = time.time()
            time_sum_spik+=(end-start)
            
            vars_used = UsedVariables(functionWasUsed)
            
            #production matrix computation
            time_start_pm =  time.time()
            PM = generateProductionMatrix(configuration)
            time_end_pm = time.time()
            time_sum_pm+=(time_end_pm-time_start_pm)

            #net gain matrix computation
            time_start_net_gain =  time.time()
            #net_gain = np.matmul(S,PM)
            
            
            num_pos = len(S)
            M = num_pos #q
            N = 4 #constant
            L = num_var #num_var
            block_size = (N,N)
            grid_size = (ceil(M/N),ceil(M/N))

            A = np.ones(M*N).reshape(M,N).astype(np.int32)
            B = np.ones(N*L).reshape(N,L).astype(np.int32)

            C = np.zeros((M, L)).astype(np.int32)

            d_a = cuda.to_device(S)
            d_b = cuda.to_device(PM)
            d_c = cuda.to_device(C)

            naive_matmul[grid_size,block_size](d_a, d_b, d_c)
            c = d_c.copy_to_host()
            net_gain = np.array(c)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            #net_gain = np.array(naiveMatrixMult(S,PM))
            time_end_net_gain = time.time()
            time_sum_ng+=(time_end_net_gain-time_start_net_gain)
            
            q_next = net_gain.shape[0]
            C_old =  np.zeros((q_next,num_var), dtype = int)

            #if variable is unused the value of the variable will be maintained
            for i in range(0,q_next):
                for j in range(0,num_var):
                    if vars_used[j] == 0:
                        C_old[i][j] = configuration[j]


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

                if ExploredStates == []:
                    nextStates.append(row.tolist())
                    node = Node(str(row.tolist()),parent = historyNode[parent])
                    historyNode.append(node)
                    continue
                else:
                    node = Node(str(row.tolist()),parent = historyNode[parent])
                    historyNode.append(node)
                    nextStates.append(row.tolist())
            
            ExploredStates.append(configuration)
            nextRemove.append(configuration)

        max_node_index = len(historyNode)-1
        min_node_index = max_node_index-len(nextStates)
        
        for state in nextRemove:
            UnexploredStates.remove(state)

        for state in nextStates:
            #if not already in ExploredStates append
            if state  not in ExploredStates:
                UnexploredStates.append(state)
            else:
                continue
            
        if (UnexploredStates == []):
            break
            
        if curr_depth < depth: 
            curr_depth += 1
            
        else:
            break
    program_end = time.time()
    
    #print("Spiking matrix time: ",time_sum)
    #print("Production matrix time: ", time_sum_pm)
    #print("Net gain time: ",time_sum_ng)
    #print("Total time: ",program_end-program_start)
    results.append([instance,time_sum_gen,time_sum_spik,time_sum_pm,time_sum_ng,program_end-program_start])

if sys.argv[1] == 'gen_ss_non_det':

    with open(r'results_GPU_nondet.txt', 'w') as fp:
        json.dump(results, fp)
    
elif sys.argv[1] == 'gen_ss_det':
    with open(r'results_GPU_det.txt', 'w') as fp:
        json.dump(results, fp)
#for pre, fill, node in RenderTree(historyNode[0]):
#    print("%s%s" % (pre, node.name))

#export tree object to Dot format for visualization
#DotExporter(historyNode[0]).to_dotfile("test.dot")
