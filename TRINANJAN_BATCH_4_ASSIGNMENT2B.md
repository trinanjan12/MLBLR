# ASSIGNMENT 2A :
**** 
  LINK FOR ASSIGNMENT 2A : 
# ASSIGNMENT 2B :
****
**Problem Statement :**
Write a python file to create the random needed to write the backprop table.

**Answer :**
Step 1: Define Input and initialize weights and biases
**code:**
``` python
# import numpy 
import numpy as np
# define the input 
X = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,0]])
#    Initialize weights and biases with random values (There are methods to initialize weights and biases but for now initialize with random values)
# wh = np.around(np.random.random((4,3)),decimals=2)
# bh = np.around(np.random.random((1,3)),decimals=2)
wh= np.array([[0.13,.01,0.1],[0.31,0.41,0.12],[0.21,0.26,0.72],[0.92,0.88,0.30]]);
bh= np.array([[0.08,0.21,0.56]])
# wout = np.around(np.random.random((3,1)),decimals=2)
# bout = np.around(np.random.random((1,1)),decimals=2)
wout = np.array([[0.26],[0.68],[0.23]])
bout = np.array([[0.48]])

print 'X =\n' , X,'\n'
print 'wh =\n' , wh,'\n'
print 'bh =\n' , bh,'\n'
print 'wout =\n' , wout,'\n'
print 'bout =\n' , bout,'\n'
```
**Input**
||||`X`    
|:--:|--|--|--|
| 0 | 1 | 0 | 0 |     
| 1 | 0 | 1 | 1 |     
| 1 | 1 | 0 | 0 |   
**wh**
|||`wh`    
|:--:|--|--|--|
| 0.13 | 0.01 | 0.10 |  
| 0.31 | 0.41 | 0.12 |     
| 0.21 | 0.26 | 0.72 |    
| 0.92 | 0.88 | 0.30 | 

**bh**
|`bh`|||    
|:--:|--|--|--|
| 0.08 | 0.21 | 0.56      

**wout**
|`wout`    
|:--:
| 0.35   
| 0.29
| 0.25    

**bout**
|`bout`    
|:--:|
| 0.48    
   

Step 2: Calculate hidden layer input:

**hidden_layer_input = matrix_dot_product(X,wh) + bh**

code:
```python
hidden_layer_input  = np.matmul(X,wh) + bh
print 'hidden_layer_input =\n',hidden_layer_input
```

**hidden_layer_input**
||||`hidden_layer_input`    
|:--:|--|--|--|
| 0.60  | 0.88 | 1.40 |     
| 1.34 | 1.36 | 1.68 |    
| 0.52 | 0.63 | 0.78 |

Step 3: Perform non-linear transformation on hidden linear input

**hiddenlayer_activations = sigmoid(hidden_layer_input)**

**code:**
```python
def sigmoid(x):
    return np.around((1/(1 +np.exp(x) ** -x)),decimals=2)
# print sigmoid(2);
hiddenlayer_activations = sigmoid(hidden_layer_input)
print 'hiddenlayer_activations =\n',hiddenlayer_activations
```

||||`hiddenlayer_activations`    
|:--:|--|--|--|
| 0.59 | 0.68 | 0.88 |     
| 0.86 | 0.86 | 0.94 |    
| 0.57 | 0.60 | 0.65 |


Step 4: Perform linear and non-linear transformation of hidden layer activation at output layer

**output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout 
output = sigmoid(output_layer_input)**

**code :**
```python
# output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout 
# output = sigmoid(output_layer_input)

output_layer_input = np.around((np.matmul(hiddenlayer_activations,wout) + bout),decimals=2)
output = sigmoid(output_layer_input);
print "output_layer_input =\n",output_layer_input
print "output=\n",output 
```
**output_layer_input**
|`output_layer_input`    
|:--:
| 1.30   
| 1.50
| 1.19  
**output**
|`output`    
|:--:
| .84   
| .9
| .8

Step 5: Calculate gradient of Error(E) at output layer

**E = y-output**

**code**
```python
# E = y-output
E = y-output;
print 'E =\n',E
```

**E**
|`E`    
|:--:
| .16   
| .1
| -.8

Step 6: Compute slope at output and hidden layer

**Slope_output_layer= derivatives_sigmoid(output)
Slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)**

**code**
```python
# define sigmoid derivative
# d(sigmoid(x))/dx = sigmoid(x) * (1-sigmoid(x))

def sigmoid_df(x) : 
    return sigmoid(x)*(1-sigmoid(x))
Slope_output_layer = sigmoid_df(output)
Slope_hidden_layer = sigmoid_df(hiddenlayer_activations)
print 'Slope_output_layer =\n',Slope_output_layer
print 'Slope_hidden_layer =\n',Slope_hidden_layer
```

**Slope_hidden_layer**
||||`Slope_hidden_layer`    
|:--:|--|--|--|
| 0.2419 | 0.2379 | 0.2176 |     
| 0.2176 | 0.2176 | 0.2059 |    
| 0.2436 | 0.2419 | 0.2400 |

**Slope_output_layer**
|`Slope_output_layer`    
|:--:
| .2211   
| .2139
| .2275

Step 7: Compute delta at output layer

**d_output = E * slope_output_layer*lr**

**code**
```python
# learning_rate is 0.1
l_rate=0.1
#d_output = E*Slope_output_layer*learning_rate 
d_output = E * Slope_output_layer * l_rate

print 'd_output =\n',d_output
```

**d_output**
|`d_output`    
|:--:
| .0.0035376   
| 0.002139
| -0.0182


Step 8: Calculate Error at hidden layer

**Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)**

**code**
```python
# Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)
Error_at_hidden_layer = np.matmul(d_output,np.transpose(wout))

print 'Error_at_hidden_layer =\n',Error_at_hidden_layer
```
**Error_at_hidden_layer**
||||`Error_at_hidden_layer`    
|:--:|--|--|--|
| 0.00091978 | 0.00240557 | 0.00081365 |     
| 0.00055614 | 0.00145452 | 0.00049197 |    
| -0.004732 | -0.012376 | -0.004186 |

Step 9: Compute delta at hidden layer

**d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer**

**code**
```python
#d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
d_hiddenlayer = np.matmul(Error_at_hidden_layer,Slope_hidden_layer)
print 'd_hiddenlayer =\n',d_hiddenlayer
```
||||`d_hiddenlayer`    
|:--:|--|--|--|
| 0.00094415 | 0.00093909 | 0.00089073     
| 0.00057088 | 0.00056782 | 0.00053857   
| -0.0048574| -0.00483135 | -0.00458254


Step 10: Update weight at both output and hidden layer

**wout = wout + matrix_dot_product (hiddenlayer_activations.Transpose, d_output) * learning_rate**


**wh = wh+ matrix_dot_product (X.Transpose,d_hiddenlayer) * learning_rate**
**code**
```python
# wout = wout + matrix_dot_product (hiddenlayer_activations.Transpose, d_output) * learning_rate
# wh = wh+ matrix_dot_product (X.Transpose,d_hiddenlayer) * learning_rate

wout = wout + np.matmul(np.transpose(hiddenlayer_activations), d_output) * l_rate
wh = wh+ np.matmul(np.transpose(X),d_hiddenlayer) * l_rate

print 'wout =\n',wout
print 'wh =\n',wh
```
**wout**
||||`wout`    
|:--:
|  0.25935527     
|  0.67933251
|  0.22932937

**wh**
||||`wh`    
|:--:|--|--|--|
| 0.12957135 | 0.00957365 | 0.0995956 |     
| 0.30960868 | 0.40961077 | 0.11963082 |   
| 0.2101515 | 0.26015069 | 0.72014293 |
| 0.92005709 | 0.88005678 | 0.30005386 |



Step 11: Update biases at both output and hidden layer

**bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate**
**bout = bout + sum(d_output, axis=0)*learning_rate**

**code**
```python
# bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate
# bout = bout + sum(d_output, axis=0)*learning_rate
bh = bh + np.sum(d_hiddenlayer, axis=0) * l_rate
bout = bout + np.sum(d_output, axis=0) * l_rate
print 'new bh = \n' ,bh
print 'new bout = \n' ,bout
```
**new bh**
|`new bh`|||    
|:--:|:--: |:--:|
| 0.07966576 | 0.20966756  | 0.55968468 |   

**new bout**
|`new bout` 
|:--:|
| 0.47874766 


