import random as rnd
import json

test_size = 5
input_size_max = 8
def generateSusbsetSumInstance(lower_bound, upper_bound, input_size):
    a = []
    for i in range(input_size):
        a.append(rnd.randint(lower_bound,upper_bound))
    b = [a]
    b.append(rnd.randint(lower_bound,upper_bound))
    return b

test_cases = []
for i in range(2,input_size_max+1):
    for j in range(test_size):
        test_cases.append(generateSusbsetSumInstance(1,20,i))
        

with open(r'test_cases.txt', 'w') as fp:
    json.dump(test_cases, fp)
       

with open(r'test_cases.txt', 'r') as fp:
    basicList = json.load(fp)
print(basicList)