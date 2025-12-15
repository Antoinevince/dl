import numpy


def dist(x, y):
    dist = 0

    for k in range(len(x)):
        dist += (y[k]- x[k])**2
    
    return (dist**0.5)





def voisins(l, k, x):
    dist_l = []
    for j in range(len(l)):
        l[j].append(dist(l[j][0], x))

    minl = l[0][2]
    minl_index = 0
    for j in range(len(l)):
        if l[j][2] < minl:
            
            minl, l[j] = l[j], l[minl_index]
            minl_index = j


    return l[k::]



def knn_local(l, k, x):
   
   lk =  voisins(l, k, x)
   etiquettedict = {}

   for j in lk:
    if not j[0] in list(etiquettedict.keys()):
        etiquettedict[j[0]] = 1
    else:
        etiquettedict[j[0]] += 1

   return max(etiquettedict)


