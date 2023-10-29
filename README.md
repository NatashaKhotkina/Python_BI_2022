# Python for Biologists 2022

Welcome!
This is my repository for home tasks in the Python course at the Bioinformatics Institute. During the course, we have studied __collections__, __PEP8__, __files__, __virtual environment__, __numpy__, __pandas__, __visuaalization__, __regular expressions__, __functional programming__, __sys__, __os__, __OOP__, __decorators__, __iterators__, __web requests__, __parallel programming__, __type annotation__, __logging__.
We have worked with different libraries, including:
+ `numpy`, `pandas`
+ `seaborn`, `matplotlib`
+ `re`
+ `sys`, `os`, `io`
+ `abc`
+ `requests`, `BeautifulSoup`
+ `typing`
+ `warnings`
+ `threading`


## Homework 1. Collections
[This tool](https://github.com/NatashaKhotkina/Python_BI_2022/tree/main/hometwork_1) can perform some basic operations with nucleic acids: 
+ __exit__ — end of program execution
+ __transcribe__ — print the transcribed sequence
+ __reverse__ — print the reversed sequence
+ __complement__ — print the complementary sequence
+ __reverse complement__ — print the reverse complementary sequence

## Homework 2. Files, FastQ filtrator.
[This script](https://github.com/NatashaKhotkina/Python_BI_2022/blob/main/homework2/fastq-filtrator.py) works with fastq files.
It can:
+ Filter reads by GC content;
+ Filter reads by their quality;
+ Filter reads by their length;
+ Save filtered and failed reads to different files.

## Homework 3. Virtual environment.
Programmer Mikhail became extremely interested in virtual environments and decided to explore them in more detail. After many years of research, he published the article and even attached the code on GitHub (https://github.com/krglkvrmn/Virtual_environment_research). Mikhail claims that anyone can easily reproduce his results. However, in practice, it turned out that Mikhail’s code cannot be run by other people. [Here](https://github.com/NatashaKhotkina/Python_BI_2022/tree/main/homework_3) you can find requirements.txt file and instructions for running Mikhail's script [pain.py](https://github.com/NatashaKhotkina/Python_BI_2022/blob/main/homework_3/pain.py).

## Homework 4. Numpy.
The utility [numpy_challenge](https://github.com/NatashaKhotkina/Python_BI_2022/blob/main/homework_numpy/numpy_challenge.py) contains several functions for matrix operations:
+ __matrix_multiplication__ performs matrix multiplication of two matrices.
+ __multiplication_check__ takes a list of matrices, and returns True if they can be multiplied in the order in which they are in the list, and False if they cannot be multiplied.
+ __multiply_matrices__ takes a list of matrices and returns the result of the multiplication if it can be obtained, or returns None if they cannot be multiplied.
+ __compute_2d_distance__ takes 2 one-dimensional arrays with a pair of values and calculates the distance between them.
+ __compute_multidimensional_distance__ takes 2 one-dimensional arrays with any equal number of values and calculates the distance between them.
+ __compute_pair_distances__ takes a 2d array and calculates a matrix of pairwise distances.

## Homework 5. Pandas and visualisation (matplotlib, seaborn)
In this homework, I practised data analysis with pandas and data visualisation with matplotlib and seaborn. 
I have written functions __read_gff__ and __read_bed6__, which read `.gff` and `.bed` files, correspondingly, and return pandas DataFrames; reconstructed `bedtools intersect` using `pandas`.
I also castommized __volcano plot__ for __DEG__ visualization - [take a look](https://github.com/NatashaKhotkina/Python_BI_2022/blob/main/pandas_and_visualization/Python_pandas_visualization.ipynb)!

## Homework 6. Regular expressions.
Here I solved different tasks, using `re.findall` and `re.sub`.

