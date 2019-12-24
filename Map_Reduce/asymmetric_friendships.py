import MapReduce
import sys

"""
Problem 4 (5 pts)
The relationship "friend" is often symmetric, meaning that if I am your friend, you are my friend. 

Implement a MapReduce algorithm to check whether this property holds. 
Generate a list of all non-symmetric friend relationships.
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: document identifier
    # value: document contents
    key = record[0]
    value = record[1]
    mr.emit_intermediate(key, ("F", value))
    mr.emit_intermediate(value, ("P", key))

def reducer(key, list_of_values):
    # key: word
    # value: list of occurrence counts
    friends = set()
    persons = set()

    for value in list_of_values:
      if value[0] == "F":
        friends.add(value[1])
      if value[0] == "P":
        persons.add(value[1])
    for p in persons:
      if p not in friends:
        mr.emit((key, p))

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)
