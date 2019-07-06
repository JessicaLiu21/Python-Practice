import timeit
start_time = timeit.default_timer() # Set time count begin 
import pandas as pd
import numpy as np
# 1.1 Data Input 
# Import the data and name its column names
# "journal" means citing journals;"reference" means cited journals;"frequence" means the number of citations between citing and cited journal. 
links_data=pd.read_csv('links.txt',sep=",",header=None,names=['Journal','Reference','Frequence'])
# Since the numbers of distinct journals and reference are different, we need to add rows to make it in a n*n matrix.
# Build a function to fill the original dataset 
def make_square_matrix(data):
    cited_set = set(data[data.columns[1]].unique())
    citing_set = set(data[data.columns[0]].unique())
    diff = cited_set.difference(citing_set) # find the different journals between reference and journal
    for i in diff: # fill the matrix 
        temp = pd.DataFrame([[i, i, 0]], columns=['Journal', 'Reference', 'Frequence'])
        data = data.append(temp, ignore_index = True)
    return data
links_df=make_square_matrix(links_data) # name the square matrix links_df
# 1.2 Creating an Adjacency Matrix
#Set the index and transform the dataset into a matrix 
links=links_df.set_index(['Journal','Reference'])
links_matrix=links.unstack('Journal') # Make the citing journals as column and cited journal as rows
# 1.3 Modifying the Adjacency Matrix
new_link=np.array(links_matrix) #transform the matrix into arrary
# Set the diagonal of Z to zero
np.fill_diagonal(new_link, 0)
# fill the nan with 0 
new_link[np.isnan(new_link)] = 0
# normalize the columns of the matrix Z
from sklearn.preprocessing import normalize
H= normalize(new_link,norm='l1',axis = 0)
H=np.mat(H) # turn H in Matrix Format
# 1.4 Identify the Dangling Nodes
col_sum=new_link.sum(axis=0)
d=[]
for i in col_sum:
    if i>0:
        d.append(0)
    else:
        d.append(1)
d=np.mat(d)
# 1.5 Calculating the Influence Vector
# Artical_Vector
distinct_journal = set(links_data['Reference'].unique())
from numpy import *
artical_initial=arange(len(distinct_journal)).reshape(len(distinct_journal),1)
#artical_initial=arange(10748).reshape(10748,1)
article_vector=np.where(artical_initial!= len(distinct_journal), 1/len(distinct_journal), artical_initial)
a=mat(article_vector)
#Initial Start Vector
initial_start=arange(len(distinct_journal)).reshape(len(distinct_journal),1)
initial_start_vector=np.where(initial_start!= 1/len(distinct_journal), 1/len(distinct_journal), initial_start)
pi=mat(initial_start_vector)
# calculate the influence vector
def calculate_eigenfactor(H,a,d,pi):
    Alpha=0.85
    Epsilon=0.00001
    k=0
    pi_last=np.ones((len(distinct_journal),1))*0
    while np.linalg.norm(pi - pi_last) >= Epsilon:
        pi_last = pi
        b=float(Alpha*np.dot(d,pi)+(1-Alpha))
        pi = Alpha*np.dot(H,pi)+np.dot(b,a)
        k += 1
    print(str(k) + " iterations")
    return pi
pi=calculate_eigenfactor(H,a,d,pi)
# 1.6 Calculate Eigenfactor 
Hpi= H.dot(pi)
EF = normalize(Hpi,norm='l1',axis = 0)
EF=EF*100
# transform the output into list
EFlist = EF.T[0].tolist()
EFlist_sort = EFlist.copy() # make a copy for the list 
EFlist_sort.sort(reverse=True) # sort the journal
top_value=EFlist_sort[:20] #print the top 20 journal
# print the index by for the value 
index=[]
for value in top_value:
    index.append(EFlist.index(value))
# store the result in a dataframe TopJournal
TopJournal= pd.DataFrame()
TopJournal['Rank'] = list(range(1,21))
TopJournal['Journal'] = index
TopJournal['Weight'] = top_value
# Timing the cell
timecount = timeit.default_timer() - start_time
print("Total Time"+str(timecount))
print(TopJournal)