import MapReduce
import sys

"""
Problem 6 (5 pts)
Assume you have two matrices A and B in a sparse matrix format, 
where each record is of the form i, j, value. 

Design a MapReduce algorithm to compute the matrix multiplication A x B
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: document identifier
    # value: document contents
    matrix = record[0]
    value = record
    if matrix == "a":
      row = record[1]
      for col in range(5):
         key = (row, col)
         mr.emit_intermediate(key, value)
    if matrix == "b":
      col = record[2]
      for row in range(5):
         key = (row, col)
         mr.emit_intermediate(key, value)


def reducer(key, list_of_values):
    # key: word
    # value: list of occurrence counts
    all_a = {}
    all_b = {}
    total = 0

    for line in list_of_values:
      if line[0] == "a":
        col = line[2]
        val = line[3]
        all_a[col] = val
      if line[0] == "b":
        row = line[1]
        val = line[3]
        all_b[row] = val

    for akey, avalue in all_a.items():
      bvalue = 0
      if akey in all_b:
        bvalue = all_b[akey]
      total += avalue * bvalue


    mr.emit((key[0], key[1], total))

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)
