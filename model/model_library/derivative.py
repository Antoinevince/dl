def get_slope(l):
    derivative_list = [((l[1][k+1]-l[1][k])/(l[0][k+1]-l[1][k])) for k in range(len(l)-1)]
    
    return derivative_list


