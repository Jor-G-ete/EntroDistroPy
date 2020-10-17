# EntroDistroPy

[![](https://img.shields.io/pypi/pyversions/EntroDistroPy.svg)](https://pypi.org/project/EntroDistro/)
[![](https://img.shields.io/pypi/l/EntroDistroPy.svg)](https://github.com/Jor-G-ete/EntroDistroPy/blob/master/LICENSE)
[![](https://img.shields.io/github/downloads/Jor-G-ete/EntroDistroPy/total)]()
[![](https://img.shields.io/github/last-commit/Jor-G-ete/EntroDistro)]()
[![](https://img.shields.io/github/v/release/Jor-G-ete/EntroDistroPy)](https://github.com/Jor-G-ete/EntroDistroPy/releases)
[![](https://img.shields.io/github/v/tag/Jor-G-ete/EntroDistroPy)]()



EntroDistroPy was created and developed thanks an idea of Enrique F. Viamontes Pernas which was my mentor and tutor and almost all the credit should go to him. This is a library which brings a collections of binners or discretiziers. Also it aims to create a whole IA model in which can be inputed a csv, after it will be analyzed, processed and converted to talk the same language as a machine learning model. Finally it would should be able to extract conclusions. The use-case which will be used in the library it's how to improve the ROI when a product is bought. Another idea in development in this library is that could be tunned and know which parameters modify to improve the column chosen or desired( in our case will be the ROI) and be tunned to obtain the maximum ROI or extract conclusion on how the parameters behave.

## Installation

### Source file

1. Download the source file from github
2. Unzip and navigate to the folder containing `setup.py` and other files
3. Run the following command: `python setup.py install`

### Pip

```python3
    pip3 install EntroDistroPy
```

## What it's entropy discretization ? Why should be used?

Entropy: How a system is untidy

Discretization or Binning: Divide a column of data in bins which have a minimum and a maximum value

Discretization or binning by entropy is state of art method :telescope::rocket:,which means that is new an experimental, the idea is that every column is binned by its own entropy which means that the binning is made using the own feature of the column. That allows to the algorithm to make a better and most fitter binning of the column.

## What's done here?

The main idea of this function is present the data as a machine learning model would wanted, for doing so a preprocess and adaption of the data must be done, the steps taken here are:

1. Standarize -> rescales data to have a mean of 0 and a standard deviation of 1 (unit variance).
2. Normalize -> rescales the values into a range of [0,1]
3. Discretize by entropy.

## How does works the entropy discretizer

1. Check that column target is inside the dataframe which is passed
2. Remove the cols which have already been discretized by that algorithm
3. Left in the pandas only the numeric type cols ( just Int or Floats)
4. Cast the pandas dataframe to a numpy array
5. Assign the X which is going to be the numpy array which will be discretizied
6. Assign the Y which is going to be the target variable extracted from the dataframe and casted to a numpy array
7. Obtain the number of features which is the same as number of cols present in the dataframe
8. Input the data into the algorithm
9. Transform one column at a time in the dataframe
10. Save the cuts,  and bin information from the discretizier
11. Save the results of the discretization.

## IA section

This section is a newbie due to it has been just added and not tested. The idea is the data outputed passed to a machine learning algorithm without any modification due to it has been done before. The model selected is [**Naive bayes**](https://scikit-learn.org/stable/modules/naive_bayes.html) depending on the data, its purpose, columns and other factors some models works better than others, for that in the same function which creates the machine learning model it has been left some test and measure to check and compare the models available.

## How is been coded the naive bayes?

1. Preprocess the Data
   1. Remove non int or float columns ( Categoricals )
   2. Remove target from the data
   3. If there are some columns discretized by entropy, hold them
   4. Remove ROI ( It was our use case) 
2. Extract the target
3. Convert the dataframe to numpy arrays and cast them as floats just in case.
4. Divide into train, test and validation
5. Deploy machine learning algorithm
6. Train
7. Predict
8. Obtain confusion matrix
9. Obtain reports on how it performed
10. Save report
11. Save confusion matrix



## Which algorithm of naive bayes has been selected

It has been selected the Multinomial Bayesian and Complemet bayesian as the main models to perform further operations . Due to with the files we have test the bayesian models, they were the models with the higher score. 

As we can see in the following image:![](./images/4.png)

## What is missing and what will be done in the next release?

The part missing is take the output of the Naive bayes and with that create a graph, for doing so, it has been selected the library **NetworkX**. The use of this library is mandatory and the further development of this library will use **NetworkX**.  



## Notes and information about the variables used

It was used a csv which can not be updated. That cvs contained a column with the name and the value of the ROI of a service or an object purchased.

### Why is selected the roi? Why did we want to improve the ROI?

The Roi is the *Return of investment*, so it's one of the most important variables. The objective among others, is to tag the ROI  and divided into different categories( bins ), once it's done that, the idea is to improve every bin or just one level if it's desired.
The model selected for doing so will be a Naives bayes machine from SciKit-learn. As it's a bayesian ( 2 options yes or no)  model we must ask ourselves a question which is **Does the ROI goes up??** 
If goes up:  [YES] -> (alto_moderado, medio_alto, medio, medio-moderado)
If goes down: [NO] -> (bajo_alto, bajo_moderado, deficiente, neutro)

*All this process is made in the function **eval_column**, which depending on the number of categories selected in the Yes and No group, it binnes by those categories. The categories goes from 0 to X ( being X the number of categories), the lower category encapsules the lowest values of the column and the higher as logic the highest, so for the YES/NO decision we will set the threshold in 5, but can be changed if desired*



## Where those who come after me must go. Recommended readings

When this library was created this method didn't exist but now with the 0.23.2 version looks like it has been implemented **Feature binnarization**:
https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-normalization 
It also could be used and recommend with a bernouilli distribution in a neural network.

![](./images/feature_bin.jpg)

## What other things can you find in this library

* Folders with old functions called *old_functions*
* Folders with old notes in spanish called *Notas,Aclaraciones y Errores*



## What's next?

The next steps as it has been stated we will be a creation of a graph using networkx. Also it will be improved the methods for the results visualization, due to evolution of matplotlib and other libraries has been left deprecated.

## Python Compatibility

* [Python](http://www.python.com) - v3.7

### Credits

This library has been created with the help of [Scikit-learn](https://scikit-learn.org/stable/), [Networkx](https://networkx.github.io/mat), [Matplotlib](https://matplotlib.org/) and [Numpy](https://numpy.org/)