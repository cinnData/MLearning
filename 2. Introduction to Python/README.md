# 2. Introduction to Python

### What is Python?

**Python** is a programming language, introduced in 1991. The current version is Python 3.10. To work with Python, you will pick an interface among the many available choices. You can have several "instances" of Python, called **kernels**, running independently in your computer.

Warning: Python is **case sensitive**. So, `type` is a Python function which returns the type of an object, but `Type` is not recognized (unless you create a new function with this name), and will return an error message.

### The Anaconda distribution

There are many distributions of Python. In the data science community, **Anaconda** (`anaconda.com`) is the favorite one. The current Anaconda distribution comes with Python 3.9. Downloading and installing Anaconda will leave you with the **Anaconda Navigator**, which opens in the browser and allows you to choose among different interfaces to Python. Alternatively, once Anaconda is installed, you can bypass the navigator through a **command-line interface** (CLI), like Terminal on Mac computers or the Anaconda prompt on Windows.

Among the many interfaces offered by Anaconda, I recommend you the **Jupyter Qt console**, which is an input/output text interface. Jupyter (Julia/Python/R) is a new name for an older project called **IPython** (Interactive Python). IPython's contribution was the IPython shell, which added some features to the mere Python language. The Qt console is the result of adding a graphical interface (GUI), with drop-down menus, mouse-clicking, etc, to the IPython shell, with a toolkit called Qt.

Part of the popularity of the IPython shell was due to the **magic commands**, which are extra commands which are written as `%cmd`. For instance, `%cd` allows you to change the **working directory**. These commands are not part of Python. Some textbooks and tutorials are still very keen on magic commands, which are occasionally mentioned in this course. To get more information about magic commands, enter `%quickref` in the console. Although, in practice, you can omit the percentage sign (so `%cd`works exactly the same as `cd`), I keep using it to distinguish the magic commands from the Python code.

Jupyter provides an alternative approach, based on the **notebook** concept. In a notebook, you can combine input, output and ordinary text. In the notebook arena, **Jupyter Notebook** is the leading choice, followed by **Apache Zeppelin**. These two are multilingual, that is, they can be used with other languages, like R, besides Python. Most data scientists prefer the console for developing their code, but use notebooks for difusion, specially for posting their work on platforms like GitHub.

Besides the Jupyter tools, Anaconda also provides a Python IDE (Integrated Development Environment) called **Spyder**, where you can manage a console and an text editor for your code. If you have previous experience with this type of interface, for instance from working with R in RStudio, you may prefer Spyder to the QtConsole.

### Python packages

Many additional resources have been added to Python in the form of **modules**. A module is just a text file containing Python code. Modules are grouped in libraries, also called **packages**, because their elements are packed according to some specific rules which allow you to install and call them together. Python can be extended by more than 300,000 packages. Some big packages, like scikit-learn, are not single modules, but collections of modules, which are then called **subpackages**.

Since the basic Python (without any package) is quite limited, you will need additional resources for practically everything. For instance, suppose that you want to do some math, and calculate the square root of 2. You will then **import** the package `math`, whose resources include the square root and many other mathematical functions. Once the package has been imported, all its functions are available. You can then apply the function `math.sqrt`. This notation indicates that `sqrt` is a function of the module `math`.

Packages are imported just for the current kernel. You finish the session by either closing the console or by restarting the kernel. You can do this with `Kernel >> Restart current Kernel` or by typing `Ctrl+.`.

With Anaconda, most packages used in this course are already available and can be directly imported. If it is not the case, you have to **install** the package (only once). There is a basic installation procedure in Python, which uses a **package installer** called `pip` (see `pypi.org/project/pip`). Using `pip` you can have a conflict of versions between packages which are related, so I would recommend you an alternative installer called `conda`, which checks your Anaconda distribution, taking care of the conflicts.

### The main packages

These notes do not look at Python as a programming language, that is, as one for developing software applications, but from a very specific perspective. Our approach is mainly based on three packages, NumPy, Matplotlib and scikit-learn.

* **NumPy** adds support for large vectors and matrices, called there **arrays**. Data sets  are managed as 2d arrays in this course.

* Based on NumPy, the library **Matplotlib** is Python's plotting workhorse.

* **scikit-learn** is a library of ML methods.

*Note*. Pandas is a library for data management, inspired in the R language. Pandas is a great tool for data management, but, unfortunately, it is slower than NumPy, so most ML courses skip it. Nevertheless, if you know Pandas, you can use it in the examples of this course, without changing the code too much. scikit-learn methods accept both NumPy arrays and Pandas objects, though they always return NumPy arrays.

### Data types

The **data types** in Python are similar to those of other languages. The type can be learned with the function `type`. Let me review the main data types:

* First, we have **integer numbers** (type `int`). There are subdivisions of this basic type, such as `int64`, but you don't need to know about that to start using Python.

* We can also have **floating-point** numbers (type `float`), that is, numbers with decimals. We also have subdivisions here, such as `float64`.

* Under type `bool`, Python admits **Boolean** values, which are either `True` or `False`. In Python, Boolean variables are converted to type `int` or `float` by applying a mathematical operator.

* Besides numbers, we can also manage **strings**, with type `str`.

* Python also has type `datetime` for dealing with **dates and times**.

### Data containers

Python has various **data container** types. The most versatile is the **list**, which is represented as a sequence of comma-separated values inside square brackets:

`mylist = ['Messi', 'Cristiano', 'Neymar', 'Coutinho']`

An element of a list is extracted indicating its place between square brackets. For instance, `mylist[1]` would extract `'Cristiano'` (in Python we start at zero). To extract a sublist with several consecutive terms, we indicate the corresponding range. For instance, `mylist[1:3]` extracts the sublist `['Cristiano', 'Neymar']` (in Python, the left limit is included but the right limit is not).

A **tuple** is like a list, represented with parentheses instead of square brackets:

`mytuple = ('Messi', 'Cristiano', 'Neymar', 'Coutinho')`

Tuples will appear sporadically in this course. Other Python data containers are the sets and the dictionaries, not used in this course (this does not imply that they are uninteresting). The packages used in data science come with new data container types, such as **NumPy arrays**. Though the elements of the Python data containers (eg lists) can have different data types, NumPy arrays have a unique type.

### Functions

Python is a fully functional language. A **function** takes a collection of **arguments**,  and returns a **value**. For instance, `len(mylist)` returns `4`. Besides the **built-in functions** like `len` and those coming in the packages that you may import, you can define your own functions. The definition will be valid only for the current kernel.

A simple example of a user-defined function would be:

`def f(x): return 1/(1 - x**2)`

Longer definitions can involve several lines of code. In that case, all the lines after the colon must be *indented*. The Jupyter interfaces create the indentation by themselves.
