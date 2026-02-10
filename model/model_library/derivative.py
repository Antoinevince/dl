def derivative_from_list_of_values(l):
    derivative_list = []
    for k in range(len(l[0])-1):
        derivative_list.append((l[0][k+1]-l[0][k])/(l[1][k+1]-l[1][k]))
    derivative_list += [derivative_list[len(l)-1]]
    return derivative_list


