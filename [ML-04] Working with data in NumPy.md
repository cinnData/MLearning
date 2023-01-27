# [ML-04] Working with data in NumPy

## The package csv

**CSV files** are text files which store data using the comma as the column separator. The names of the columns typically come in the first row, and every other row corresponds to a data point. If Excel is installed in your computer, the files with the extension `.csv` are associated to Excel (with a specific Excel icon).

There are many ways to import/export data from/to a CSV file in Python. The classic approach is based on the package `csv`, included in the standard library.  As usual, you start by importing the package:

```
import csv
```

The built-in function `open` creates a connection to a text file, assuming that the path specified makes sense in your computer. The mode of the connection can be `mode='r'` (read), `mode=‘w'` (write) or others. This note shows how to use these two basic modes to import/export data from/to CSV files in Python.

## Importing data from CSV files

To import data from a CSV file, you start by opening a connection to the source file. You have to complete the name of the file with the corresponding path, unless it is in the **working directory**. In Jupyter Qt Console, this is `/Users/username` in Mac computers and `C:/Users/username` in Windows computers. In a Jupyter notebook, the working directory is the folder where the notebook is. Take care of using the slash (`/`), not the backslash, (`\`) to separate the folders in the path.

```
conn = open('fname.csv', mode='r')
```

*Note*. The default encoding of the reading mode of open is `encoding='utf-8'`. You don’t have to worry about that, unless you import text data, containing special characters, from a file created in a Windows computer. This situation is very rare in machine learning.

Next, you create a **CSV reader**: 

```
reader = csv.reader(conn)
```

The default **delimiter** for the reader is the comma, but you can use `delimiter=';'` for CSV files created by Excel apps that use the semicolon as the column separator. The function `list` will write the contents of the source file as a list in which every item is a list containing the data from a row of the source file. All the data are imported as type `str`.

```
data = list(reader)
```

Now you can close the connection:

```
conn.close()
```

Practitioners pack all this as:

```
with open('fname.csv', mode='r') as conn: 
    reader = csv.reader(conn)
    data = list(reader)
```

If the column names come in the first row, which is typical, it is recommended to put them in a separate list, to have them at hand:

```
header = data[0]
```

Now, If all the data are numeric, which is typical in machine learning, you can get an numeric 2D array with:

```
X = np.array(data[1:]).astype(float)
```

In most cases, you will not include all the columns from the source file directly in a matrix `X`. Depending on the analysis you plan, you may organize the data as you need for the planned analysis. This is better understood in specific examples.

## Exporting data to CSV files

To export a data set with the package csv, you write your data set as a list in which every item is a list which contains the data from a row of the data set, putting the column names as the first row. Let us suppose that the data array is `X`, and the column names are collected in the list header. You pack this as the new list data:

```
data = [header] + list(X)
```

Note that I write `[header]`, not `header`, as the first item. If you don’t have a header, you can make things simpler using directly `X`, without creating any list.

Now, you open (and close) a connection to a file `filename.csv,` with the appropriate path and use a **CSV writer** to fill it. Be careful here: if the file does not exist, the connection will create a new, empty file, and the writer will write the data there. But if the file already exists, the writer will overwrite it.

```
with open('fname.csv', mode='w', newline='', encoding='utf-8') as conn: 
    writer = csv.writer(conn)
    writer.writerows(data)
```

The argument `newline=''` is needed in Windows but not in Macintosh. If you omit it, a Windows computer will put extra blank lines between rows. If not specified, the encoding used by the new file is platform-dependent (meaning UTF-8 in Mac and various things in Windows, depending on your region). You can also use `delimiter=';'` if it suits you.
