import input_ss

def genDetSubsetSumFunctionLocationMatrix(input_size):
    L=[]
    for i in range(input_size+2**(input_size+1)):
        lst = [0] * int(input_size+2**(input_size+1))
        lst[i] = 1
        L.append(lst)
    return L

def genDetSubsetSumFunctionMatrix(input_size):
    F=[]
    zeros = [0] * int(input_size+2**(input_size+1))
    
    #Function matrix for input neurons
    for i in range(input_size):
        tempList = zeros.copy()
        tempList[i] = 1
        F.append(tempList)
        
    #Function matrix for sums of inputs 
    for i in range(input_size, (input_size+1)+(2**input_size) - 1):
        tempList = zeros.copy()
        tempList[i] = -1
        F.append(tempList)
        
    #Function matrix equality checking neurons
    for i in range((input_size+1)+(2**input_size)-1, (input_size+1)+(2**(input_size+1))-1 ):
        tempList = zeros.copy()
        tempList[i] = 1
        F.append(tempList)  
        
    return F
    
def genDetSubsetSumNoFunc(input_size):
    zeros = [0] * int(input_size+2**(input_size+1))
    zeros[(input_size+2**(input_size+1))-1] = 1
    return zeros
    
def genDetSubsetThresholdMatrix(input_size):

    zeros = [0] * int(input_size+2**(input_size+1))
    T = zeros.copy()
    has_Threshold = zeros.copy()
    
    
    for i in range(input_size+1, (input_size+1)+(2**input_size) - 1 ):
        T[i] = -1
        has_Threshold[i] = 1
        
        
    for i in range((input_size+1)+(2**input_size) - 1, (input_size+1)+(2**(input_size+1)) - 1):
        T[i] = 1
        has_Threshold[i] = 1
        
        
    return T,has_Threshold

def powerset(fullset):
  listsub = list(fullset)
  subsets = []
  for i in range(2**len(listsub)):
    subset = []
    for k in range(len(listsub)):            
      if i & 1<<k:
        subset.append(listsub[k])
    subsets.append(subset)        
  return subsets
 
def genDetSubsetSynapseList(input_size):

    syn = []
    num_combi = (2**input_size) - 1
    temp_set = []
    
    for i in range(input_size):
        temp_set.append(i)
    
    subsets = powerset(temp_set)
    subsets.remove([])
    
    k = 0
    for subset in subsets:
        for j in range(len(subset)):
            syn.append((subset[j],k+input_size+1))
        k+=1
 
        
    for i in range(num_combi):
        syn.append((input_size,i+input_size+1))
        syn.append((i+input_size+1,i+input_size+1+num_combi))
        syn.append((i+input_size+1+num_combi,input_size - 1 + 2**(input_size+1)))
    
    for i in range(num_combi):
        syn.append((input_size,i+input_size+1))
        syn.append((i+input_size+1,i+input_size+1+num_combi))
        syn.append((i+input_size+1+num_combi,input_size - 1 + 2**(input_size+1)))
    return syn
    
    
       
def genConfiguration(S,SUM):
    zeros = [0] * int(input_size+2**(input_size+1))
    C = zeros.copy()
    for i in range(len(S)+1):
        if i < len(S):
            C[i] = S[i]
        else:
            C[i] = SUM+1
    return C
    

set = input_ss.set #input set of numbers to the subset sum problem
target_sum = input_ss.target_sum
input_size = len(set)


C = genConfiguration(set,target_sum)
L = genDetSubsetSumFunctionLocationMatrix(input_size)
F = genDetSubsetSumFunctionMatrix(input_size)
no_func = genDetSubsetSumNoFunc(input_size)
T,has_threshold = genDetSubsetThresholdMatrix(input_size)
syn = genDetSubsetSynapseList(input_size)
