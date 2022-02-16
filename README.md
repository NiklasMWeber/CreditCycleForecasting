# Signature Regression and an Application to Credit Cycle Forecasting

This GitHub repository delivers the implementation of the Singature regression and all simulations metioned in my M.Sc. Thesis, which can be found in the repository as well. The explanations here might not be sufficient to understand every detail. It is expected that the reader is familiar with the contets of my thesis, especially with Chapter 5 explaining the numerical experiments. The code reuses some classes and ideas from A. Fermanian's work in [1], but there are also many original contributions.

## The code

The code consitst of several files:

- The `train.py` file contains the implementation of the Signature regression, it contains the method to select a runcation order via cross-validation and a method to compute the signatures of a set of training data.
- The `tools.py` file contains some auxiliary functions that are used in the other remaining files now and then.
- The `dataGeneration.py` provides classes that return different types of data, either synthetically generated or using the real world macroeconomic data from the macrodata.npy array.
- The `expermiments.py` file contains as class that performs a simulation with synthetically generated data. It compares singature regression and linear regression. The exact configuration can be adjusted in the if __name__ == '__main__' part of the script.
- The `experimentsCCF.py` file contains a similar class as the experiments.py file. It can be used to perform simulations with the macroeconomic data and the varied parameters are different from the simulation with synthetically generated data.
- The `createPlots.py` file can be used to create all plots shown in the thesis, after all experminents are conducted.
- The `macrodata.npy` array contains the macroeconomic data for the Credit-Cycle-Forecasting experiment. It is an easy way to import the macrodata without the need of downloading the data from the different sources.
- The `requirements.txt` can be used to create an python environment containing all necessary packages.

To run an experiment with synthetically data, you need to specify a configuration in the if`configurations.py` and then run the script `main.py` with arguments the name of the configuration and the number of iterations. For example, if you want to do one run of a linear regression on signatures, with a constant `Kpen=1`, on simulated sinus data of dimension 2, you can use the configuration

```python
if __name__ == '__main__':
    ##########################################################################
    ### Specify experiment configuration
    nameForExperimentFolder = 'exp1'
    comparer = CompareSigAndLinReg(testRatio = 0.5)
    
    nPathsList = [33, 50, 100, 200, 500, 1000]
    numForPartitionList = [3,5,10,20,50,100]
    dimPath = 3
    trueM = 5
    plotTrue = False
    iterations = 20
    
    G = dg.GeneratorFermanian1(dimPath = dimPath, nPaths = nPathsList[0], num = numForPartitionList[0], trueM = trueM)
    
    MSE_Sig_testMatrix, MSE_LinReg_testMatrix, R_Sig_testMatrix, R_LinReg_testMatrix = \
        comparer.createComparisonMatrix(nPathsList,numForPartitionList,G, iterations= iterations, mHat = None, addTime = True, addBase = True)
    
    mHat_Matrix = comparer.mHat_Matrix
    ##########################################################################
```

and then run the file.

```
python experiments.py
```
The results of each experiment are stored in the `exp` directory. They can be loaded and plottet running the `createPlots.py` file.

The main arguments in a configuration are:
* nameForExperimentFolder: here you specify the name of the folder, where the results will be stored.
* comparer: An custom made class that performs the comparison between signature regression and linear regression for all combinations of `nPats` and `num` (in the thesis `nPoints`).
* testRatio: The test ratio determines, which ratio of the data is used for testing the performance. The reamining set will be used for training.
* nPathsList: The list of `nPaths` that will be considered by the `comparer` class.
* numForPartitionList: The list of `num` that will be considered by the `comparer` class.
* dimPath: the dimension of the synthetically generated paths.
* trueM: true truncation order for some types of synthetically generated data. This variable is ignored in the cases where choosing a true truncation order is not necessary.
* plotTrue: boolean flag. If True the script will create plots of the results.
* iterations: indicates how many times the comparison between signature regression and linear regression will be repeated for every combination of `nPaths` and `num`.
* G: the `dataGenerator`that creates the data for every iteration according to the input `dimPath`, `nPaths`,`num` and possibly `trueM`.

To run an experiment on real-world macro data with the similar `experimentsCCF.py` class the procedure is almost the same, however, some variables differ:

```python
if __name__ == '__main__':
    ##########################################################################
    ### Specify experiment configuration
    nameForExperimentFolder = 'exp4'
    comparer = CompareSigAndLinReg(testRatio = 0.33)

    windowSizes = [3,4,6,8,12,16]
    forecastingGaps = [0,1,2,3,4,5,7,9,11,15,19,23]
    trueM = None
    plotTrue = False
    iterations = 20
    
    G = dg.GeneratorMacroData(windowSize = 3, forecastGap = 0)
    
    MSE_Sig_testMatrix, MSE_LinReg_testMatrix, R_Sig_testMatrix, R_LinReg_testMatrix = \
        comparer.createComparisonMatrix(windowSizes = windowSizes, forecastingGaps = forecastingGaps, dataGenerator = G,
					iterations = iterations, mHat = None, addTime = True, addBase = True)

    mHat_Matrix = comparer.mHat_Matrix
    ##########################################################################
```

The main differences are:
* windowSizes: replaces nPaths. It determines the size of windows that are used fore forecasting the response.
* forecastingGaps: replaces numForPartitionList. It determines the time horizon of the forecast. 0 corresponds to forecasting the first response directly after the window, 1 to forecasting the second response and so on...

## Reproducing the numerical experiments

We give below the steps to reproduce the results of the thesis.

### Environment

All the necessary packages may be set up by running
`pip install -r requirements.txt`

### Data

All the data will either be generated or can be found in the macrodata.npy file. If it seems to appropriate to download updated data just download the data from the sources outlined in the thesis and update the paths to the data in the `importData()` method in tools.

### Running the experiments

Run `experiments.py` file three times with the above configurations, only changing nameForExperimentFolder and G:
1. nameForExperimentFolder = 'exp1', G = dg.GeneratorFermanian1
2. nameForExperimentFolder = 'exp2', G = dg.GeneratorFermanianIndependentMax
3. nameForExperimentFolder = 'exp3', G = dg.GeneratorFermanianGaussian

Then run `experimentCCF.py` with the above configuration and
4. nameForExperimentFolder = 'exp4', G = dg.GeneratorMacroData


The `createPlots.py`file can be utilized to create the corresponding plots (or set `plotTrue = True` to obtain them right away).

## Citation

[1]:
```bibtex
@article{fermanian2021linear,
  title={Linear functional regression with truncated signatures},
  author={Fermanian, Adeline},
  journal={arXiv:2006.08442},
  year={2021}
}
```

