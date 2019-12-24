import MapReduce
import sys

"""
Problem 4:

Consider a set of key-value pairs where each key is sequence id and each value is a string of nucleotides,
 e.g., GCTTCCGAAATGCTCGAA....

Write a MapReduce query to remove the last 10 characters from each string of nucleotides, 
then remove any duplicates generated.

"""

mr=MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: document identifier
    # value: document contents
    key = record[0]
    value = record[1]
    mr.emit_intermediate(value[:-10], "")

def reducer(key, list_of_values):
    # key: word
    # value: list of occurrence counts

    mr.emit(key)

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)