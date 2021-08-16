# 2. Introduction to Python

### What is Python?

**Python** is a programming language, introduced in 1991. The current version is Python 3.9. To work with Python, you will pick an interface among the many available choices. You can have several "instances" of Python, called **kernels**, running independently in your computer.

Python is **case sensitive**. So, `type` is a Python function which returns the type of an object, but `Type` is not recognized (unless you create a new function with this name), and will return an error message.

### The Anaconda distribution

There are many distributions of Python. In the data science community, **Anaconda** (`anaconda.com`) is the favorite one. The current Anaconda distribution comes with python 3.8 and contains most of the resources used in this course. Downloading and installing Anaconda will leave you with the **Anaconda Navigator**, which opens in the browser and allows you to choose among different interfaces to Python. Alternatively, once Anaconda is installed, you can bypass the navigator through a **command-line interface** (CLI), like Terminal on Mac computers or the Anaconda prompt on Windows.

Among the many interfaces offered by Anaconda, I recommend you the **Jupyter Qt console**, which is an input/output text interface. Jupyter (Julia/Python/R) is a new name for an older project called **IPython** (Interactive Python). IPython's contribution was the IPython shell, which added some features to the mere Python language. The Qt console is the result of adding a graphical interface (GUI), with drop-down menus, mouse-clicking, etc, to the IPython shell, by means of a toolkit called Qt.

Part of the popularity of the IPython shell was due to the **magic commands**, which are extra commands which are written as `%cmd`. For instance, `%cd` allows you to change the home directory. These commands are not part of Python. Some textbooks and tutorials are still very keen on magic commands, but others prefer to skip them. They are mentioned occasionally in this course. To get more information about magic commands, enter `%quickref` in the console. Although, in practice, you can omit the percentage sign (so `%cd`works exactly the same as `cd`), I keep using it to distinguish the magic commands from the Python code.

Jupyter provides an alternative approach, based on the **notebook** concept. In a notebook, you can combine input, output and ordinary text. In the notebook arena, **Jupyter Notebook** is the leading choice, followed by **Apache Zeppelin**. These two are multilingual, that is, they can be used with other languages, like R, besides Python. Most data scientists prefer the console for developing their code, but use notebooks for difusion, specially for posting their work on platforms like GitHub.

Besides the Jupyter tools, Anaconda also provides a Python IDE (Integrated Development Environment) called **Spyder**, where you can manage a console and an text editor for your code. If you have previous experience with this type of interface, for instance from working with R in RStudio, you may prefer Spyder to the QtConsole.

### Python packages

Many additional resources have been added to Python in the form of **modules**. A module is just a text file containing Python code (extension `.py`). Modules are grouped in libraries, also called **packages**, because their elements are packed according to some specific rules which allow you to install and call them together. Python can be extended by more than 300,000 packages. Some big packages, like scikit-learn, are not single modules, but collections of modules, which are then called **subpackages**.

Since the basic Python (without any package) is quite limited, you will need additional resources for practically everything. For instance, suppose that you want to do some math, and calculate the square root of 2. You will then **import** the package `math`, whose resources include the square root and many other mathematical functions. Once the package has been imported, all its functions are available. So, you can apply the function `math.sqrt`. This notation indicates that `sqrt` is a function of the module `math`.

Packages are imported just for the current kernel. You finish the session by either closing the console or by restarting the kernel. You can do this with `Kernel >> Restart current Kernel` or by typing `Ctrl+.`.

If you work with the Anaconda Python distribution, most packages used in this course are already available and can be directly imported. If it is not the case, you have to **install** the package (only once). There is a basic installation procedure in Python, which uses a **package installer** called `pip` (see `pypi.org/project/pip`). Using `pip` you can have a conflict of versions between packages which are related, so I would recommend you an alternative installer called `conda`, which checks your Anaconda distribution, taking care of the conflicts. Both `pip` and `conda` can be run from the console as a magic commands (`%pip`, `%conda`).

### The main packages

These notes do not look at Python as a programming language, that is, for developing software applications, but from a very specific perspective. Our approach is mainly based on three packages, NumPy, Matplotlib, Pandas and scikit-learn.

* **NumPy**, released in 1995, is a library adding support for large vectors and matrices, called there **arrays**.  

* Based on NumPy, the library **Matplotlib** is Python's plotting workhorse.

* **scikit-learn**, released in 2008, is a library of machine learning methods.

*Note*. Pandas is a library for data management, inspired in the R language. Pandas is a great tool for data management, but, unfortunately, slower than NumPy, so most ML courses skip it. Nevertheless, if you know Pandas, you can use it in the examples of this course. scikit-learn methods accept both NumPy arrays and Pandas objects, though they always return NumPy arrays.

### Learning about Python

There are many books for learning about Python, but some of them would not be appropriate for learning how to work with data in Python. It can even happen that you do not find anything about data in many of them. Mind that Python has so many uses that the intersection of the know-how of all Python users is relatively narrow. For an introduction to Python as a programming language, in a computer science context, I would recommend Zelle (2010). For the self-learning data scientist, McKinney (2017) and VanderPlas (2017) are both worth their price.

There is also plenty of learning materials in Internet, including MOOC's. For instance, **Coursera** has a pack of courses on Python (see `coursera.org/courses?query=python`). But the most attractive marketplace for data science courses is **DataCamp**. They offer, under subscription or academic license, an impressive collection of courses, most of them focused on either R or Python. In addition to follow DataCamp courses, you can also benefit from the **DataCamp Community Tutorials**, which are free and cover a wide range of topics. Finally, a good place to start is `learningpython.org`.

### Data types

The **data types** in Python are similar to those of other languages. The type can be learned with the function `type`. Let me review the main data types:

* First, we have **integer numbers** (type `int`). There are subdivisions of this basic type, such as `int64`, but you don't need to know about that to start your Python trip.

* We can also have **floating-point** numbers (type `float`), that is, numbers with decimals. We also have subdivisions here, such as `float64`.

* Under type `bool`, Python admits **Boolean** values, which are either `True` or `False`. In Python, Boolean variables are converted to type `int` or `float` by applying a mathematical operator.

* Besides numbers, we can also manage **strings**, with type `str`.

* Python also has type `datetime` for dealing with **dates and times**.

### Data containers

Python has various **data container** types. The most versatile is the **list**, which is represented as a sequence of comma-separated values inside square brackets:

`mylist = ['Messi', 'Cristiano', 'Neymar', 'Coutinho']`

An element of a list (or a tuple) is extracted indicating its place between square brackets. For instance, `mylist[1]` would extract `'Cristiano'` (in Python we start at zero). To extract a sublist with several consecutive terms, we indicate the corresponding range. For instance, `mylist[1:3]` extracts the sublist `['Cristiano', 'Neymar']` (in Python, the left limit is included but the right limit is not).

A **tuple** is like a list, represented with parentheses instead of square brackets:

`mytuple = ('Messi', 'Cristiano', 'Neymar', 'Coutinho')`

Other Python data containers are the sets and the dictionaries, not used in this course (this does not imply that they are uninteresting). The packages used in data science come with new data container types, such as **NumPy arrays**. Though the elements of the Python data containers (eg lists) can have different data types, NumPy arrays have consistency constraints.

### Functions

Python is a fully functional language. A **function** takes a collection of **arguments**,  and returns a **value**. For instance, `len(mylist)` returns `4`. Besides the **built-in functions** like `len` and those coming in the packages that you may import, you can define your own functions. The definition will be forgotten when the session is closed, so you have to include the definition in your code.

A simple example of a user-defined function would be:

`def f(x): return 1/(1 - x**2)`

Longer definitions can involve several lines of code. In that case, all the lines after the colon must be *indented*. The Jupyter interfaces create the indentation by themselves.

### Loops and conditional logic

**Loops** are used in practically all computer languages, so, if you have seen them if you have any kind of experience in that context. In particular, `for` loops are used to avoid repetition. Suppose that you wish to extract the first letter of the names of the list `mylist`, storing them in a new list. You can use the `for` loop to iterate the extraction:

`inilist = [name[0] for name in mylist]`

This would return `['M', 'C', 'N', 'C']`. Loops are much less frequent in the machine learning, because NumPy and scilit-learn provide **vectorized functions**, that, when applied to an array such as a Pandas series, return an array with same shape, whose terms are the values of the function on the corresponding terms of the original array. Nevertheless, we may use occasionally a `for` loop in this course.  

Also ubiquitous in programming is **conditional logic**, operationalized through **if-then-else** commands. You also have this in Python. For instance, if you wish to create a dummy flag for names with more than 5 letters in the list `mylist`, you can do it with:

`flaglist = [1 if len(name) > 5 else 0 for name in mylist]`

This would return `[0, 1, 0, 1]`. It is also rare to find explicit if-the-else arguments in machine learning, since "vectorial" syntax is preferred (and typically leads to a faster execution).
