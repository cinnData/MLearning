## Tutorial - Introduction to Python ##

# Typing Python code #
2 + 2
a = 2 + 2
a
b = 2 * 3
b - 1
b**2

# Python packages #
import math
math.sqrt(2)

# Numeric types #
type(a)
b = math.sqrt(2)
type(b)
type(2)
type(2.0)
float(2)
int(2.3)
round(math.sqrt(3), 3)
math.floor(math.sqrt(3))
math.ceil(math.sqrt(3))

# Boolean variables #
d = 5 < a
d
type(d)
a == 4
math.sqrt(d)
1 - d

# Strings #
c = 'Messi'
type(c)

# Lists #
mylist = ['Messi', 'Cristiano', 'Neymar', 'Coutinho']
len(mylist)
newlist = mylist + [2, 3]
newlist
len(newlist)
mylist[0:2]
mylist[3:]
mylist[:3]

# Functions #
def f(x):
    y = 1/(1 - x**2)
    return y
f(2)
f(1)
f('Mary')
def g(x, y): return x*y/(x**2 + y**2)
g(1, 1)
