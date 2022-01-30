# 3. NumPy and Matplotlib

### NumPy arrays

In Mathematics, a **vector** is a sequence of numbers, and a **matrix** is a rectangular arrangement of numbers. Operations with vectors and matrices are the subject of a branch of mathematics called linear algebra. In Python (and in many other languages), vectors are called one-dimensional (1d) **arrays**, while matrices are called two-dimensional (2d) arrays. Arrays of more than two dimensions can be managed in Python without pain.

Python arrays are not necessarily numeric. Indeed, vectors of dates and strings appear frequently in data science. In principle, all the terms of an ordinary array must have the same type, so the array itself can have a type, although you can relax this constraint in structured arrays, to appear later in this course. 

The usual way to import NumPy is:

`import numpy as np`

Arrays can be created directly from lists with the NumPy function `array`. Two simple examples:

`arr1 = np.array(['a', 'b', 'c', 'd'])`

`arr2 = np.array([[2, 3, 1], [7, -3, 2.6]])`

An array has a collection of attributes, such as `ndim`, `shape` and `dtype`. They are extracted as `arr.attr`. For instance, `arr1.shape` would return `(5,)`, while `arr2.shape` will return `(2,3)`.

### NumPy functions

NumPy incorporates vectorized forms of the **mathematical functions** of the package `math`. A **vectorized function** is one that, when applied to an array, returns an array with the same shape, whose terms are the values of the function on the corresponding terms of the original array. For instance, the square root function `np.sqrt` takes the square root of the terms of a numeric array.

NumPy also provides common **statistical functions**, such as `mean`, `max`, `sum`, etc.

### Subsetting arrays

Subsetting works in 1d arrays as in lists. For instance, `arr1[1:3]` would extract an array containing `'b'` and `'c'`. The same applies to 2d arrays, but we need two indexes within the square brackets. The first index selects the rows, and the second index the columns.

Subarrays can also be extracted by means of expressions. For instance, the first row of `arr2` can be extracted with:

`arr2[arr2[:, 1] > 0, :]`

### Plotting with Matplotlib

**Matplotlib** has an impressive range of graphical methods, including image processing. As many other libraries in the Python world, Matplotlib has several API's, which confounds the beginners. In this context, an **application programmers interface** (API) is like an idiom that you speak to call the functions of the library. It defines the kinds of requests that can be made and how to make them. 

Matplotlib offers you a choice between two API's, the pyplot API, used in this course, and the object-oriented API. This course uses the **pyplot API**. Beware that, if you use Google or similar ways to find information about plotting in Matplotlib, the solutions found can come in any of the two API's. This can make Matplotlib to look more difficult than it really is.

The subpackage `matplotlib.pyplot` is a collection of command style functions that make Matplotlib work like MATLAB. It is typically imported as:

`import matplotlib.pyplot as plt`

Each `pyplot` function makes some change to a figure, such as changing the default size, adding a title, plotting some lines in the plotting area, decorating the plot with labels, etc. It is mainly intended for simple cases of programmatic plot generation and for interactive plots.
