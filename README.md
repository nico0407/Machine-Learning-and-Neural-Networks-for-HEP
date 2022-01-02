# Uni_project

Collection of different Machine Learning models and neural network structures applied to a particle dataset using Keras, Tensorflow and Sklearn.
The aim of the project is to focus on how to make the better choice of a suitable ML algorithm for the job at hand.
Complexity and learning curve analyses are part of the visual analytics tools that help for comparing the merits of various ML algorithms.


# Table of Contents
1. [About_the_project](#About)
    1. [Dataset](#Dataset)
3. [Implementation](#Implementation)
    1. [ML model comparison](#MLmodelcomparison)
    2. [ML comparison variating training dataset size](#MLcomparisonvariatingtrainingdatasetsize)
    3. [Neural Network performance](#NeuralNetworkperformance)
    4. [Neural Network performance variating training dataset size](#NeuralNetworkperformancevariatingtrainingdatasetsize)


## About the project <a name="About"></a>

### Dataset <a name="Dataset"></a>
The Dataset used, 'pid-5M' is a dataset availabkle online and downloaded from Kaggle's dataset page. (https://www.kaggle.com/naharrison/particle-identification-from-detector-responses)

That's a simplified dataset of a GEANT based simulation for electron-proton inelastic scattering measured by a particle detector system.
It simulates in the final state four particle types with an id number associated - positron (-11), pion (211), kaon (321), and proton (2212); six detector responses. Some detector responses are zero due to detector inefficiencies or incomplete geometric coverage of the detector.
Is composed of 5000000 rows and 7 columns:
+ ID
+ Momentum(GeV/c)
+ Theta angle(rad)
+ Beta value
+ Number of photoelectrons
+ Inner energy(GeV)
+ Outer energy(GeV)

Some photos attached


## Implementation <a name="Implementation"></a>
A juppiter notebook was used for a better manipolation of the script. Due to the possibility to run the code piece by piece it was possible to do more test on the models accuracy without run everytime the whole file given the huge structure of this one.
The library used are:
+ Pandas for manipolation of the dataframe
+ Numpy
+ Matplotlib
+ Seaborn
+ Sklearn for the ML model implementation
+ Keras
+ Tensorflow

The aim of the project was to discriminate pions respect to the other particles, so first of all was done a data manipulation on the id values, the pion id was changed into 1 and the other particles'(electrons, protons and kaons) id into 0.
Then given the extreme number of rows in first analysis were considered just the first 50000 ones.

In such a way the dataframe was more usable for a machine learning implementation, in addition the dataframe was splitted into an 'x' and 'y' part. The latter one is the id column whose role is as a target to understand is the classification is done in the right way or not, instead the otherone whas inside all the other six remanent columns.
Then was implemented a data splitting into test and train with a test_size equal to 0.30 and after that an other one, dividing the test dataset into a validation and a test one. That was done giving them the same dimension.

### ML model comparison <a name="MLmodelcomparison"></a>


### ML comparison variating training dataset size <a name="MLcomparisonvariatingtrainingdatasetsize"></a>

### Neural Network performance <a name="NeuralNetworkperformance"></a>

### Neural Network performance variating training dataset size <a name="NeuralNetworkperformancevariatingtrainingdatasetsize"></a>




# Table of contents
1. [Introduction](#introduction)
2. [Some paragraph](#paragraph1)
    1. [Sub paragraph](#subparagraph1)
3. [Another paragraph](#paragraph2)

## This is the introduction <a name="introduction"></a>
Some introduction text, formatted in heading 2 style

## Some paragraph <a name="paragraph1"></a>
The first paragraph text

### Sub paragraph <a name="subparagraph1"></a>
This is a sub paragraph, formatted in heading 3 style

## Another paragraph <a name="paragraph2"></a>
The second paragraph text



# H1
## H2
### H3
#### H4
##### H5

Alt-H1
======

Alt-H2
------

Lists
1. First ordered list item
2. Another item
⋅⋅* Unordered sub-list. 
1. Actual numbers don't matter, just that it's a number
⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behaviour, where trailing spaces are not required.)

* Unordered list can use asterisks
- Or minuses
+ Or pluses
