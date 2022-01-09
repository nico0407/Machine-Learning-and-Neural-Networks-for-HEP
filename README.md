# Uni_project

Collection of different Machine Learning models and neural network structures applied to a particle dataset using Keras, Tensorflow and Sklearn.
The aim of the project is to focus on how to make the better choice of a suitable algorithm.
Complexity and learning curve analyses are part of the visual analytics tools that help for comparing the various algorithms.


# Table of Contents
- [About_the_project](#About)
    - [Dataset](#Dataset)
- [Implementation](#Implementation)
    - [ML model comparison](#MLmodelcomparison)
    - [ML comparison variating training dataset size](#MLcomparisonvariatingtrainingdatasetsize)
    - [Neural Network performance](#NeuralNetworkperformance)
    - [Neural Network performance variating training dataset size](#NeuralNetworkperformancevariatingtrainingdatasetsize)
- [Conclusion](#Conclusion)

## About the project <a name="About"></a>

### Dataset <a name="Dataset"></a>
The Dataset used, 'pid-5M' is a dataset available online and downloaded from Kaggle's dataset page. (https://www.kaggle.com/naharrison/particle-identification-from-detector-responses)

That's a simplified dataset of a GEANT based simulation for electron-proton inelastic scattering measured by a particle detector system.
It simulates in the final state four particle types with an id number associated(positron (-11), pion (211), kaon (321), and proton (2212)) and six detector responses. Some detector responses are zero due to detector inefficiencies or incomplete geometric coverage of the detector.
Is composed of 5000000 rows of values and 7 columns:
+ ID
+ Momentum(GeV/c)
+ Theta angle(rad)
+ Beta value
+ Number of photoelectrons
+ Inner energy(GeV)
+ Outer energy(GeV)

Here are plotted the distribution of the six detector responses.

![alt text](https://github.com/nico0407/Uni_project/blob/main/images/Data_visualisation/Data_hinsto.png)

Is also reported the following plot feaguring the beta value(v/c) of a particle against the momentum of this one.

![alt text](https://github.com/nico0407/Uni_project/blob/main/images/Data_visualisation/Data_betavsmomentum.png)

It's easy to see how in this plot the pion trace is very different from the kaon and the proton one. This different behaviour is due to the different values of masses particles, and so, for the same values of the momenta, the beta value is different.
Particular difficulties is between the discrimination though electron and pion, in that case also evidentiate by the lack of statistics.

The rest of the images concerning dataset analysis are left into the specific folder of data visualization.

## Implementation <a name="Implementation"></a>
A juppiter notebook was used for a better manipolation of the script. Due to the possibility to run the code piece by piece it was possible to do more test on the models accuracy without run everytime the whole file given the huge structure of this one.
The library used are:
+ Pandas
+ Numpy
+ Matplotlib
+ Seaborn
+ Sklearn
+ Keras
+ Tensorflow

The aim of the project was to discriminate pions respect to the other particles, so first of all was done a data manipulation on the id values, the pion id(211) was changed into 1 and the other particles'id into 0 (for proton, electrons and kaons), and given the extreme number of rows, in first analysis were considered just the first 50000 values.

In such a way the dataframe was more usable for a machine learning implementation, in addition the dataframe was splitted into an 'x' and 'y' part. The latter one is the id modified column that plays the role of the target to quantify if the training is done in the right way or not, instead in the other one(x) there were all the other six remanent columns.
Then was implemented a data splitting into test and train with a test_size equal to 0.30. Soon after, an other splitting was done, dividing the test dataset into a validation and a test one, both with the same dimension. The data subset proportions is the following:

training : validation : testing = 70 : 15 : 15

The validation set is a set of data, separate from the training set, that is used to validate our model performance during training.
The model is trained on the training set, and, simultaneously, the model evaluation is performed on the validation set after every epoch.
The main idea of splitting the dataset into a validation and test set is to prevent our model from overfitting i.e., the model becomes really good at classifying the samples in the training set but cannot generalize and make accurate classifications on the data it has not seen before. 

The test set is a separate set of data used to test the model after completing the training.

The project was divided in two main part. The first one focused on the use of machine learning algoritms, and the second one instead has inside the implementation of a neural network model. 

------

### ML model comparison <a name="MLmodelcomparison"></a>
The models utilized were the following:
+ Decision Tree Classifier, with both gini and entropy criterion
+ Ada Boost Classifier
+ Logistic Regression
+ K neighbors Classifier
+ SGDC Classifier
+ Rnadom Forest Classifier

For some models, a variation of hyperparameters was implemented, and after each trial the learning curves were plotted, a red one representing the learning curve for the training dataset, and in blue the validation one.
The learning curve represent the variation of the accuracy of the model in function of some paramethers(e.g. the minimum sample leaf or the maximum depth in the case of a decison tree classifier)

varying maximum depth                                                                                                |  varying minimum sample leaf
:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------:
![alt](https://github.com/nico0407/Uni_project/blob/main/images/model_comparison/DTCgini_variating_max%20depth.png)  |  ![alt](https://github.com/nico0407/Uni_project/blob/main/images/model_comparison/DTCgini_variating%20min_sample_leaf.png)


Confronting the two curves one can check if, at the end of the time of training, the model predict well the data samples and which choose of hyperparameter could be the most suitable one.

In principle one can have cases of overfitting, when the model approximate too well the training set, and has a low predictive power in any other possible data sample. A way to avoid that is by looking at the behaviour of the learning curves of training and validation. If the accuracy on the training set starts to became higher than the one on the validation set and the discrepancy between the two do not vanish, it can be a case of overfitting. Also the test set play an important role in understanding the goodness of the model.

After have fitted the models to the training dataset this were tested in making predictions over the test dataset trough the predict mode of the models. In addition also functions recall_score, '''precision_score''' and f1_score were used to compute the accuracy, togheter with the cross validation functions of sklearn, such as cross_val_predict and cross_val_score. Cross validation is a procedure that divides a dataset into k non overlapping folds. Each of the folds is given an opportunity to be used as test set, while all other folds collectively are used as a training dataset. A total of k models are fit and evaluated on the k hold-out test sets and the mean performance is reported.

For each model udsed is also provided the ROC curve with the relative computation of the area under the curve(auc). In the following are reported an example.

Ada boost                                                                                               |  DTC
:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------:
![alt](https://github.com/nico0407/Uni_project/blob/main/images/model_comparison/ROC_curves/ROC_AB.png)  |  ![alt](https://github.com/nico0407/Uni_project/blob/main/images/model_comparison/ROC_curves/ROCDTCentropy.png)

At the end, the accuracy score of every model was computed via the function "accuracy_score" and also via the "f1_score", the results are reported below.
Was choosen to use also the latter one because combine inside two other metrix such as recall_score and precision_score, so the result of f1_score will be high only if both recall and precision are high.
 


accuracy_score                                                                                                  |  f1_score
:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------:
![alt](https://github.com/nico0407/Uni_project/blob/main/images/model_comparison/comparison/compare1.png "left")  |  ![alt](https://github.com/nico0407/Uni_project/blob/main/images/model_comparison/comparison/compare2.png "right")

-----

### ML comparison variating training dataset size <a name="MLcomparisonvariatingtrainingdatasetsize"></a>

The project also present an other comparison between models. In this case was applied a modification in the dataset available for training each model. Specifically this one was setted to zero and then implemented by adding a fraction of 10% of the data at the time, and at each step was calculated and plotted the learning curve for the training and validation data set, until the model reach the 100% of the data fed for training. 
For semplicity is reported just an example of this, the other images are available in the specific repository section.

Moreover at each step it has been computed the ROC curve relative to the percentage of the training data fed.
Also in that case is presented just one model, the rest are available in the images folder section of the repository.



Learning curve                                                                                                  |  ROC
:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------:
![alt](https://github.com/nico0407/Uni_project/blob/main/images/model_comparison/training_variations/knn10.png)  |  ![alt](https://github.com/nico0407/Uni_project/blob/main/images/model_comparison/ROC_curve_training_changing/ROC_RFC10.png)

--------

### Neural Network performance <a name="NeuralNetworkperformance"></a>
The study was done also using neural network models.

Firstly was built a model with no hidend layer but the results were not good, in particular there was a pretty high discrepancy between training accuracy and the testing one. 

Sequently the following functions were defined:

+ A function for building the model. In particular this provides one hidden layer, and each layer was interprised by dropout layer. This function allows to choose values of the learing rate, the dropout probability, the number of neurons for both the input and the hidden layer and even the choice for the optimizer. At the end it also compile the model.
+ An other function was built for running the model. Has the fit function inside and provide the arbitrary choose of the number of epochs to run and the batch size. Moreover make a checkpoint, saving the model values step by step. While the model is fitted on the training data at the same time is also validated with the validation data set. At the end are ploted together learning curves for training and validation, both for the accuracy and for the loss.
+ In the last function is implemented the testing for the model with the "evaluate" function that provide a score of the model on the testing data set. That's much useful for determine if a model overfit or underfit, if has enough predictive power.

Here is reported an example of the Adam optimizer with the corrispective ROC curve.

ACCURACY                                                                                                  |  LOSS |ROC
:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------:|:--:|
![alt](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/first_models/Adam50.png "left")  |  ![alt](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/first_models/Adam50loss.png "right") | ![alt](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/first_models/Adam50ROC.png)



After some trial of esecution, was done a comparison between models built with different oprimizer, both with the use of the evaluate function, described in the testing section above, and with the f1_score function. Is evident the discrepancy between the SGD optimizer model and the other two.

![alt text](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/accuracy/Accuracy1.png)

Moreover was applied a change contemporary in the number of neurons for each layer and for the number of epochs of run using the Adam optimizer. In that case an improvement in the learning curve shows up as the number of epochs of run increase. Can be seen in the following gif, where the plots are refered to a number of epoch of run respectevely of 5,10 and 25, while the variation of the number of neurons is in the x-axis.

![alt text](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/Variating_numepoch_and_neurons/5zvq2b.gif)

The neural network built with Adam optimizer works well in prediction. Good values of testing score was given in output, in accorrding with the training one, the model do not overfit and seems to have a good predictive power. The accuracy of training and testing are in agreement with an accuracy of 97%.

Even though the one hidden layer neural network(also known as multilayer perceptron(MLP)) works well in prediction, is there also the implementation of an other Neural Network to see if it can improve the performance of the previous one, composed with two hidden layer devided by a drop out one.
Also here it's been done a study on the better value for the hyperparameters, testing the model in variation of neuron number for each layer and the number of epoch of run. That model was executed for a growing number of epoch, until it came up into overfitting of training data.

It has been runned for 50, 100, 200, 400 epochs and after the 100nt one the values of training accuracy and testing starts to separate one from the other.
Specifically the training score starts to became higher than the validation one, moreover also the score on the test dataset is pretty lower respect to the testing one. That's a case of overfitting of the model.

That's why in that case the model need to be symplified up to a number of epoch of run not higher than 100. For such values of hyperparamethers training and test score are almost the same and the accuracy of the model is up to 97.4%

ACCURACY                                                                                                  |  LOSS |ROC
:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------:|:--:|
![alt](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/Sequence/Seq400.png "left")  |  ![alt](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/Sequence/Seq400loss.png "right") | ![alt](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/Sequence/Seq400ROC.png)

ACCURACY                                                                                                  |  LOSS |ROC
:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------:|:--:|
![alt](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/Sequence/Seq100.png "left")  |  ![alt](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/Sequence/Seq100loss.png "right") | ![alt](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/Sequence/Seq100ROC.png)

--------

### Neural Network performance variating training dataset size <a name="NeuralNetworkperformancevariatingtrainingdatasetsize"></a>

The variation of the available training size has been done also for the neural network section of the project, specifically only for the MLP.

The usable training set was changed, and starting from zero, it was increased by a number of step of 10% of the data available.
For each step was done the computation and the conseguent plot of validation and training curve, and the construction of the ROC curve. Concerning the latter ones is evident how it improves increasing the trainig data set, that goes from 10% to 100% of the data available, leaving validation and testing data sets unaltered.

Learning curves | ROC
:-------:|:----------:|
![alt](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/variation_training_set/plot3.png)    | ![alt](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/variation_training_set/cinque.png)


After this trial then, the fraction of the total data available for training, testing, and validation was changed simultaneously, manteining the proportions decided at the beginning the same(training : validation : testing = 70 : 15 : 15).

So in that case is not only varying the training set, but all the available amount of data for building the model.

Learning curves | ROC
:-------:|:----------:|
![alt](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/variation_training_set/l.png)    | ![alt](https://github.com/nico0407/Uni_project/blob/main/images/NN_model/variation_training_set/i.png)

Is apreciable an increasing of accuracy with the increasing of the percentage of the data available, starting with a low score in the initial step due to the too low statistiscs, reaching a plateau in higher percentage. Therefore the choice of cutting the whole datset up to just 50000 rows was a quite good approximation.

Moreover in that final case are plotted together training, validation and testing scores, as shown in the plot legend. The values of the three scores remain similar after every changing step.

## Conclusion <a name="Conclusion"></a>
The study done was aimed to discriminate which could be the best way to classify pion from the other particles in this specific dataset.
Many attempts have been made using machine learning models and neural network models varying hyperparameters and computing the efficiency of every model trying to avoid overfitting. 

This was done using tools like cross-validation techniques, implementign dropout layers in the neural network, using testing and validation datasets and trying to simplify the model when necessare.

After many trial and variations of the hyperparamethers of the various model, the ones that seems to work better are the Random forest classifier and the Adaboost, they give the highest value of the f1_score, so contemporaneusly a pretty high value of recall and prediction at the same time. Respectevely with accuracy of 97.5% and 97.3%.

Regarding the neural network implementation, is appreciable to see how the one layer model do not work enough well on discrimination, given the poor value of testing on the model, respect to the training one. Instead the multilayer perceptron works well, and has appropiate learing curves with enough good result in training and validation accuracy and also in testing score. The best one in this case is the model built with the Adam optimizer, with an accuracy level of 97%.

The last model with two hidden layer instead seems to work well in accuracy score up to 100 epochs of run, reporting a value of accuracy of 97.4%.
After the 100th epoch of run approximatevely the model seems to overfit.
