ex1=[8,[4,[2],[3]],[3,[1],[1,[1],[1]]]]

ex2=['D',['B',['A'],['C']],['F',['E'],['H',['G'],['I']]]]

# constructor 
def tree(label,branches=[]):
    for branch in branches:
        assert is_tree(branch)
    return [label]+list(branches)

# selector 
def label(tree):
    return tree[0]

def branches(tree):
    return tree[1:]

def is_tree(tree):
    if type(tree) != list or len(tree)<1:
        return False
    for branch in branches(tree):
        if not is_tree(branch):
            return False
    return True 




def count_nodes(t):
    if is_leaf(t):
        return 1
    total=0
    for b in branches(t):
        total += count_nodes(b)
    return total+1


def collect_leaves(t):
    if is_leaf(t):
        return [label(t)]
    lst=[]
    for b in branches(t):
        lst += collect_leaves(b)
    return lst 

collect_leaves(ex2)
