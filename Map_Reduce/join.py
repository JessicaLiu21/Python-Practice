import MapReduce
import sys

"""
Your MapReduce query should produce the same result as this SQL query executed against an appropriate database.

SELECT * 
FROM Orders, LineItem 
WHERE Order.order_id = LineItem.order_id

You can consider the two input tables, Order and LineItem, 
as one big concatenated bag of records that will be processed by the map function record by record.
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: document identifier
    # value: document contents
    key = record[1]
    value = record

    mr.emit_intermediate(key, value)

def reducer(key, list_of_values):
    # key: word
    # value: list of occurrence counts
    
    orderlist=list_of_values[0]
    for line in list_of_values:
        if line[0] != 'order':
            new=[]
            for i in orderlist:
                new.append(i)
            for j in line:
                new.append(j)
            
            mr.emit(new)
# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)
