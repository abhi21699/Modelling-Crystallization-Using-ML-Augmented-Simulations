import numpy as np
#from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
y_test = []

def distance(a,b):
    return ((float(a[0])-float(b[0]))**2 + (float(a[1])-float(b[1]))**2 + (float(a[2])-float(b[2]))**2)**0.5

def read_file(filename, start_line, end_line):
    data = []
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, 1):
            if line_number < start_line:
                continue
            if line_number > end_line:
                break
            columns = line.strip().split()
            if len(columns) >= 3:
                row = columns[-3:]
                data.append(row)
    return data

state = []
state_eigen_val = []


filename = './test_crystal.xyz'  # Replace with the actual filename

j = 1
final_list = []
start_line = 3
end_line = 1
while (end_line < 4360):
    end_line = (216 * j) + 2*j
    result = read_file(filename, start_line, end_line)
    final_list.append(result)
    start_line = end_line + 3
    j += 1
#print(len(final_list))
for i in final_list:
    matrix = np.zeros((108,108),dtype=float)
    #print(len(i))
    #print(matrix[0][106])
    for j in range(108):
        k = 108
        for k in range(108,216):
            matrix[j][k-108]= distance(i[j],i[k])
    eigen_val , eigen_vect = np.linalg.eig(matrix)
    a = sorted(eigen_val)
    #state_eigen_val.append([a[-1],a[-2],a[-3]])
    state_eigen_val.append([a[-1],a[-2],a[-3],a[-4],a[-5],a[-6],a[-7],a[-8],a[-9],a[-10]])
    state.append('Crystalline')
    y_test.append(0)
    '''# ---------------------REDUCTION OF EIGEN VALUES (CRYSTAL)-------------------------------------------------------------

    covariance_matrix = np.cov(eigen_vect.T)
    eigenvalues_pca, eigenvectors_pca = np.linalg.eig(covariance_matrix)

    # Sort the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues_pca)[::-1]
    eigenvalues_pca_sorted = eigenvalues_pca[sorted_indices]
    eigenvectors_pca_sorted = eigenvectors_pca[:, sorted_indices]

    # Select the top 2 eigenvalues and eigenvectors
    compressed_eigenvalues = eigenvalues_pca_sorted[:2]
    compressed_eigenvectors = eigenvectors_pca_sorted[:, :2]

    # Project the original eigenvectors onto the compressed eigenvectors
    compressed_data = np.dot(eigen_vect, compressed_eigenvectors)
    state.append("Crystalline")
    state_eigen_val.append(compressed_eigenvalues)
    #print(compressed_eigenvalues)
    #print(matrix)
    '''

filename = './test_liq.xyz'  # Replace with the actual filename

j = 1
final_list = []
start_line = 3
end_line = 1
while (end_line < 4360):
    end_line = (216 * j) + 2*j
    result = read_file(filename, start_line, end_line)
    final_list.append(result)
    start_line = end_line + 3
    j += 1
#print(len(final_list))
for i in final_list:
    matrix = np.zeros((108,108),dtype=float)
    #print(matrix[0][106])
    for j in range(108):
        k = 108
        for k in range(108,216):
            #print(len(i))
            #print(k)
            matrix[j][k-108]= distance(i[j],i[k])
            #print(k)
            #k+=1
            #pass
            #print(j, k)
    eigen_val , eigen_vect = np.linalg.eig(matrix)
    a = sorted(eigen_val)
    #state_eigen_val.append([a[-1],a[-2],a[-3]])
    state_eigen_val.append([a[-1],a[-2],a[-3],a[-4],a[-5],a[-6],a[-7],a[-8],a[-9],a[-10]])
    state.append('Liquid')
    y_test.append(1)

    #print(eigen_val)
    #print(eigen_vect)
    '''
    # ---------------------REDUCTION OF EIGEN VALUES (CRYSTAL)-------------------------------------------------------------

    covariance_matrix = np.cov(eigen_vect.T)
    eigenvalues_pca, eigenvectors_pca = np.linalg.eig(covariance_matrix)

    # Sort the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues_pca)[::-1]
    eigenvalues_pca_sorted = eigenvalues_pca[sorted_indices]
    eigenvectors_pca_sorted = eigenvectors_pca[:, sorted_indices]

    # Select the top 2 eigenvalues and eigenvectors
    compressed_eigenvalues = eigenvalues_pca_sorted[:2]
    compressed_eigenvectors = eigenvectors_pca_sorted[:, :2]

    # Project the original eigenvectors onto the compressed eigenvectors
    compressed_data = np.dot(eigen_vect, compressed_eigenvectors)
    state.append("Liquid")
    state_eigen_val.append(compressed_eigenvalues)
    #print(compressed_eigenvalues)
    #print(matrix)
#print(state_eigen_val)
#print(state)
'''
# x = []
# y = []
# z = []
# for i in state_eigen_val:
#     x.append(i[0].real)
#     y.append(i[1].real)
#     z.append(i[2].real)

# #print(x)
# #print(y)
# for i in range(len(x)):
#     if state[i] == "Liquid":
#         label = 'Liquid'
#         color = 'blue'
#     else:
#         label = 'Crystalline'
#         color = 'red'
#     plt.text(x[i],y[i],z[i],label,color=color)
# plt.plot(x,y,z)
# plt.show()
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(x)):
    if state[i] == "Liquid":
        label = 'Liquid'
        color = 'blue'
    else:
        label = 'Crystalline'
        color = 'red'
    ax.scatter(x[i], y[i], z[i], c=color, label=label)

ax.set_xlabel('Eigenvalue 1')
ax.set_ylabel('Eigenvalue 2')
ax.set_zlabel('Eigenvalue 3')

plt.legend()
plt.show()
'''
x_test = []
for i in state_eigen_val:
    a = []
    a.append(i[0].real)
    a.append(i[1].real)
    a.append(i[2].real)
    a.append(i[3].real)
    a.append(i[4].real)
    a.append(i[5].real)
    a.append(i[6].real)
    a.append(i[7].real)
    a.append(i[8].real)
    a.append(i[9].real)
    x_test.append(a)
# print(x_test)
# #print(state)
# print(y_test)






