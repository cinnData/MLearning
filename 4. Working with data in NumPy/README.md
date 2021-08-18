# 4. Working with data in NumPy

### Arrays with mixed type

Real data come with **mixed data type**. NumPy has a special data type, called `object`, which helps to deal with mixed data types. To see how this works, take the data matrix:

`arr = np.array([['Trump', 2016], ['Obama', 2008]])`

NumPy converts the numeric column to string type, to have a common type, so `arr` has `dtype='<U5>'`, meaning a string of maximum length 5. But suppose that you specify `dtype=object`:

`mix_arr = np.array([['Trump', 2016], ['Obama', 2008]], dtype=object)`

Now, the array `mix_arr` has a different type, which encapsulates the original data types. These original types can be recovered by extracting the columns separately and specifying the appropriate data type for each one. For instance, to get the President names:

`np.array(mix_arr[:, 0], dtype='str')`

### Structured arrays

Besides using the data type `object`, NumPy has an alternative approach to mixed data types, based on **structured arrays**. Let us consider a variation of the above example:

`str_arr = np.array([('Trump', 2016), ('Obama', 2008)], dtype=[('president', 'str'), ('year', 'int')])`

`str_arr` is a structured array, that is a 1d array whose data type is a composition of simpler types organized as a sequence of named fields. Each term in the structured array stands for a row in a tabular data set. The different pieces of information contained in a structured array can be extracted with Pythonic syntax:

* Rows are extracted as in a 1d array: `str_arr[0]` is the first row, while `str_arr[-1]` is the last row. A range of rows is managed as usual.

* Columns can be extracted by name: `str_arr['president']` is the first column, and `str_arr['year']` is the second column. With a list of column names, you can extract a structured subarray.

* Individual entries can be extracted with two indexes: `'Trump'` is `str_arr[0][0]`, and `'Obama'` is `str_arr[0][0]`.  

### Converting structured arrays

Structured arrays can be converted to unstructured arrays, and conversely. In particular, in this course we import data from text files as structured arrays, converting them to  unstructured arrays to be accepted by scikit-learn methods. There is a special function for this job, the function `structured_to_unstructured` of the subpackage `lib.recfunctions`. The syntax is simple:

`from numpy.lib.recfunctions import structured_to_unstructured`

`arr = structured_to_unstructured(str_arr)`

Note that `structured_to_unstructured` will return an array with a unique data type. So, you may have to take care of the data types (or to select the columns) before applying it-

### Importing data from text files to NumPy arrays

Python can import and export data sets in tabular form in various ways and using various file formats. This chapter explains how to import data from a **CSV file** to a NumPy array. CSV files are text files which use the comma as the column separator. The names of the columns typically come in the first row, and every other row corresponds to a data point.

If Excel is installed in your computer, the files with the extension `.csv` are associated to Excel (so, they have a special Excel icon). Although CSV is a very popular format, these files must be handled with care when they contain string data, specially when these strings can contain commas and line breaks. In this course, even if we may find some columns with string type in a data set, those columns will not create a problem.

The NumPy function `genfromtxt` imports data from a text file to a structured array. The syntax could be:

`data = np.genfromtxt(filename, delimiter=',', names=True, dtype=None, encoding='utf-8')`

A brief comment on the arguments:

* You have to complete the name of the file with the path, unless it is in the home directory. In a Jupyter console, this folder is the home folder of the current user in that computer, which is `/Users/username` in Mac computers and `C:\Users\username` in Windows computers. In a Jupyter notebook, the home directory is the folder where the notebook is. You can also import data from a remote file, using the corresponding URL.

* `genfromtxt` imports data from text files with various delimiters, so you have to specify that the delimiter is the comma. If you use European CSV files, replace the comma by the semicolon.

* The argument `names=True` tells Python that the first row of the file comes with the column names. This is the usual practice.

* The argument `dtype=None` tells Python that it has to guess the data type. The default is type `float` for all the columns. You can also specify the type of every column, which makes Python read the data faster.

* The `encoding` argument is only needed in the recent versions of NumPy. The choice of the encoding system is relevant only when your string data contains special characters (such as ñ, or á).

Python guesses the data type from the content as follows. When all the entries in a column are numbers, that column is imported as numeric. If there is a decimal point in any of the entries, the type is `float64` (called here `f8`). If there is no decimal point, the type is `int64` (called `i8` here). If at least one of the entries is not numeric, the column has type `str` (called `U` here).

### Sorting data

You can **sort** 1d arrays with the NumPy function `sort`, which sorts the terms of the array in ascending order. The syntax is:

`np.sort(arr)`

The same function can be used to sort the rows of a structured array. The syntax would be, now:

`np.sort(str_arr, order='colname')`

### Unique values

The NumPy function `unique` returns the unique values of an array. When applied to an unstructured array, it returns a 1d array. When applied to a structured array, it returns a structured array with the unique rows (duplicate rows are dropped).

In the first case, with an additional argument, you can count the occurrences of every value. The syntax would be:

`np.unique(arr, return_counts=True)`

The function `unique` would return now two objects, separated by a comma.
