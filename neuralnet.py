from numpy import array, dot, random
import numpy as np
import math

def sigmoid(x):
    return 1.0/(1 + math.exp(-x))

f = open('digits.wdep', 'r')
lines = f.readlines()
f.close()

imagesOf1 = []
image = ''
for line in lines:
    if len(line.strip('\n')) == 32:
        image += line + '\n'
    else:
        number = line.strip('\n').strip()
        if number == '1' or number == '2' or number == '3':
            tempImage = image.split('\n')
            tempImage = filter(None, tempImage)
            tempImageArray = []
            for i in tempImage:
                arrays = list(i)
                tempImageArray.append(arrays)
            imagesOf1.append(tempImageArray)
            image = ''
        else:
            image = ''

weights = random.rand(65, 10)
hidden_weights = random.rand(10,3)
expectedFor1 = array([0, 0, 1])
expectedFor2 = array([0, 1, 0])
expectedFor3 = array([0, 1, 1])
eta = 0.2

#for _ in range(1000):
for image in imagesOf1:
    new_matrix = [[0 for i in range(8)] for j in range(8)]
    rows = 0
    columns = 0
    for i in range(0, 32, 4):
        for j in range(0, 32, 4):
            average = 0
            for k in range(i, i + 4):
                for l in range(j, j + 4):
                    average += int(image[k][l])
            new_matrix[rows][columns] = average/16.0
            columns += 1
        rows += 1
        columns = 0
    for i in range(8):
        for j in range(8):
            print new_matrix[i][j],
        print
    print '----------------------------------------------------------------'
    newImageArray = array([new_matrix[i][j] for i in range(8) for j in range(8)] + [1])
    print newImageArray
    
    netj = dot(newImageArray, weights)
    yj = []
    for i in netj:
        yj.append(sigmoid(i))
    netk = dot(array(yj), hidden_weights)
    zk = []
    for i in netk:
        zk.append(sigmoid(i))
    tk_zk = expectedFor1 - array(zk)
    fdashofnetk = []
    for i in netk:
        fdashofnetk.append(sigmoid(i)*(1 - sigmoid(i)))
    deltak = np.multiply(tk_zk, array(fdashofnetk))

    sigma = dot(hidden_weights, deltak)
    
    fdashofnetj = []
    for i in netj:
        fdashofnetj.append(sigmoid(i)*(1 - sigmoid(i)))
    sumy = np.sum(dot(hidden_weights, deltak.T))
    deltaj = np.multiply(sumy, fdashofnetj)
    
    #print np.shape(deltaj.T)#, np.shape(newImageArray.T)
    weights += eta * dot(newImageArray[np.newaxis].T, deltaj[np.newaxis])
    #print np.shape(deltak), np.shape(yj)
    print yj, np.shape(yj[np.newaxis])
    print deltak[np.newaxis].T, np.shape(deltak)
    
    hidden_weights += eta * np.asscalar(dot(deltak[np.newaxis], yj[np.newaxis]))
    break
