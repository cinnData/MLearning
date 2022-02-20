# Cheatsheet - NumPy

### Slicing and indexing

* `arr[n]`: for a 1d array `arr`, returns the term of index `n`. Since Python starts counting at zero, this would be the entry at the `n-1`-th place.

* `arr[n:m]`: for a 1d array `arr`, returns a subarray containing the terms of `arr` whose indexes go from `n` to `m-1`. If `n` is missing, it is assumed to be equal to `0`. If `m` is missing, it is assumed to be equal to `len(arr)`.

* `arr[n,m]`: for a 2d array `arr`, returns the term at row `n` and  column `m`.

* `arr[r1:r2, c1:c2]`: for a 2d array, returns the subarray resulting from selecting the rows from `r1` to `r2-1` and the columns from `c1` to `c2-1`. Omissions are understood as for 1d array.

* `arr[bm]`: for a boolean mask `bm`, returns a subarray of `arr`containg the terms for which `bm` takes value `True`. 

### Unstructured arrays

* `arr.astype()`: converts `arr` to a different data type.

* `np.cumsum(arr)`: returns a 1d array containing the cumulative sums of the terms of `arr`. For a 2d array, cumulative sums are calculated row following row. With argument `axis=0`, returns a 2d array containing the cumulative sums calculated separately for every column. With argument `axis1`, separately for every row. The same as `arr.cumsum()`.

* `arr.dtype`: `returns` the data type of `arr`. If the elements of `arr` are literals, it will be `int64`, `float64`, `<Ul` or `bool`. More complex data can have data type `object`.

* `arr.ndim`: returns the number of dimensions, 1 for a 1d array and 2 for 2d array.

* `arr.shape`: returns the shape of the array `arr`, as a tuple. For a 1d array, it is `(l,)`, and, for a 2d array, `(r,c)`.

* `np.abs(arr)`: replaces the terms of a numeric array by their absolute values. The same as `abs(arr)`.

* `np.array(lst)`: for a list `lst` whose elements are literals of the same data type, it returns a 1d array containing the same elements as `lst`. For a list whose elements are lists of the same length, containing literals of the same data type, it returns a 2d array.

* `np.argsort(arr)`: takes a 1d array `arr` and returns a 1d array containing the index that every term will take if `arr` were sorted in ascending order.

* `np.concatenate([arr1, arr2, ...], axis=n)`: concatenates a list of arrays vertically (`axis=0`) or horizontally (`axis=1`). The default is `axis=0`. When `axis=0`, all the arrays must have the same number of columns, and, when `axis=1`, the same number of rows.

* `np.corrcoef([arr1, arr2, ...])`: returns the correlation matrix a list of numeric 1d arrays of the same length. For two arrays, the square brackets can be omitted. It can also take a 2d array, returning the correlations of the row vectors (not of the columns).

* `np.int64(arr)`: converts a `float` or `bool` array to data type `int64`. The same can be done with `arr.astype('int')`.

* `len(arr)`: for a 1d array, returns the number of terms of `arr`. For a 2d array, the number of rows.

* `np.linspace(a,b,n)`: returns a 1d array with `n` equally spaced terms, starting by `a` and ending by `b`.

* `np.max(arr)`: returns the maximum of the terms of `arr`. With argument `axis=0`, returns a 1d array containing the maximums of the columns of `arr`. With argument `axis1`, the maximums of the rows. The same as `arr.max()`.

* `np.mean(arr)`: returns the mean of the terms of `arr`. With argument `axis=0`, returns a 1d array containing the column means. With argument `axis1`, the row means. The same as `arr.mean()`.

* `np.min(arr)`: returns the minimum of the terms of `arr`. With argument `axis=0`, returns a 1d array containing the minimums of the columns of `arr`. With argument `axis1`, the minimums of the rows.  The same as `arr.min()`.

* `np.round(arr, d)`: rounds the terms of a numeric array to a specified number of digits. The same can be obtained with `arr.round(d)`.

* `np.reshape(arr, sh)`: changes the shape of an array. The number of terms of the reshaped array must be equal to that of the original array. The same can be done with `arr.reshape(sh)`.

* `np.sort(arr)`: sorts a 1d array in ascending order. To reverse this, add `[::-1]`. For higher dimensional arrays, look at the manual.

* `np.sum(arr)`: returns the sum of the terms of `arr`. With argument `axis=0`, returns a 1d array containing the column totals. With argument `axis1`, the row totals.  The same can be obtained with `arr.sum()`.

* `np.transpose(arr)`: transposes a 2d array. The same as `arr.transpose()` and `arr.T`.

* `np.unique(arr)`: returns a 1d array containing the unique terms of `arr`, in ascending order. With additional argument `returns_counts=True`, it returns a second array containing the number of occurrences of every unique value.

### Structured arrays

* `np.genfromtxt(fname, delimiter=',', names=True, dtype=None, encoding='utf-8')`: imports data from a CSV file to a structured array, taking the column names from first row. For variations, look at the manual.

* `structured_to_unstructured(arr)`: converts a structured array to an unstructured (ordinary) array. The subpackage `numpy.lib.recfunctions` must be already imported.
