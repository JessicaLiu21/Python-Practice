def funky(x):
    if x<0:
        print('negatory')
    elif x%2 == 0:
        return "I can't even"
    elif x%3 == 0:
        return "3 of a kind"


def f(x):
    i=0
    total=0
    while i<x:
        i=i+1
        total=total+i

    return i,total

def fib(n):
    pred,curr=0,1
    k=1
    while k<n:
        pred,curr=curr,pred+curr 
        k=k+1
    return curr

def area(r,shape_constant):
    assert r>0, 'A length should be positive'
    return r*r*shape_constant

def area_square(r):
    return area(r,1)

def area_circle(r):
    return area(r,pi)

def area_heaxgon(r):
    return area(r,3*sqrt(3)/2)


from math import pi,sqrt
def sum_natural(n):
    """sum the first N natural numbers
    >>> sum_natural(5)
    15
    """
    total, k = 0, 1 
    while k<=n:
        total,k = total+k, k+1
    return total 


## returning a function using it own name 
def print_sums(n):
    print(n)
    def next_sum(k):
        return print_sums(n+k)
    return next_sum

print_sums(1)(3)(5)