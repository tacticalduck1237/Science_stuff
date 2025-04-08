import os, sys, inspect
import pyeq3
import numpy 
import numpy as np
import scipy
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt

functionString = 'A*X**(-B)*exp(-K/X)'

# note that the constructor is passed the function string here
equation = pyeq3.Models_2D.UserDefinedFunction.UserDefinedFunction(inUserFunctionString = functionString)

# copy and paste resercher data here:
data = '''
  С(НП)        Дыхание
618	4.96165545184686
618	3.87438104993684
618	5.08548908550028
618	4.58626635627558
618	2.238049386774
618	4.33225255927944
618	3.69295217920564
618	2.20982436048086
618	1.96509579693051
268.695652173913	3.58316185795834
268.695652173913	3.85951692131691
268.695652173913	3.58309934791739
268.695652173913	4.42098290302848
268.695652173913	2.88922537978515
268.695652173913	2.8896884267614
268.695652173913	4.47236053656647
268.695652173913	5.02409462315979
268.695652173913	3.86555426937174
123.6	2.88621679624643
123.6	4.74273079143705
123.6	3.65839410913888
123.6	3.03780779592167
123.6	2.78106021288475
123.6	3.0739001322161
123.6	3.21612805248714
123.6	3.61108796483794
123.6	3.35583721275324
79.741935483871	3.2605190283314
79.741935483871	2.64712801093714
79.741935483871	3.30772855621509
79.741935483871	3.82324647264087
79.741935483871	3.40316765262595
79.741935483871	3.26226855164811
79.741935483871	3.28512511313107
79.741935483871	3.59930358062502
79.741935483871	3.96391854044065
59.0822179732314	2.72765318247987
59.0822179732314	2.83483412878562
59.0822179732314	2.91752851887773
59.0822179732314	3.21953739120251
59.0822179732314	2.88370816183583
59.0822179732314	3.21660745939985
59.0822179732314	3.78314890956175
59.0822179732314	2.48281009905462
59.0822179732314	2.91689219011828
29.0550070521862	3.3257839937866
29.0550070521862	2.94577267457829
29.0550070521862	3.22973297284088
29.0550070521862	3.69017847438639
29.0550070521862	3.02593731843251
29.0550070521862	3.22510487501078
29.0550070521862	3.33566860130779
29.0550070521862	1.60389004281371
29.0550070521862	3.15545275479714

'''
pyeq3.dataConvertorService().ConvertAndSortColumnarASCII(data, equation, False)
equation.Solve()


##########################################################


print("Equation:", equation.GetDisplayName(), str(equation.GetDimensionality()) + "D")
print("Fitting target of", equation.fittingTargetDictionary[equation.fittingTarget], '=', equation.CalculateAllDataFittingTarget(equation.solvedCoefficients))
print("Fitted Parameters:")

for i in range(len(equation.solvedCoefficients)):
    print("    %s = %-.16E" % (equation.GetCoefficientDesignators()[i], equation.solvedCoefficients[i]))


equation.CalculateModelErrors(equation.solvedCoefficients, equation.dataCache.allDataCacheDictionary)
print()
for i in range(len(equation.dataCache.allDataCacheDictionary['DependentData'])):
    print('X:', equation.dataCache.allDataCacheDictionary['IndependentData'][0][i],)
    print('Y:', equation.dataCache.allDataCacheDictionary['DependentData'][i],)
    print('Model:', equation.modelPredictions[i],)
    print('Abs. Error:', equation.modelAbsoluteError[i],)
    if not equation.dataCache.DependentDataContainsZeroFlag:
        print('Rel. Error:', equation.modelRelativeError[i],)
        print('Percent Error:', equation.modelPercentError[i])
    else:
        print()
print()


##########################################################


equation.CalculateCoefficientAndFitStatistics()

if equation.upperCoefficientBounds or equation.lowerCoefficientBounds:
    print('You entered coefficient bounds. Parameter statistics may')
    print('not be valid for parameter values at or near the bounds.')
    print()

print('Degress of freedom error',  equation.df_e)
print('Degress of freedom regression',  equation.df_r)

if equation.rmse == None:
    print('Root Mean Squared Error (RMSE): n/a')
else:
    print('Root Mean Squared Error (RMSE):',  equation.rmse)

if equation.r2 == None:
    print('R-squared: n/a')
else:
    print('R-squared:',  equation.r2)

if equation.r2adj == None:
    print('R-squared adjusted: n/a')
else:
    print('R-squared adjusted:',  equation.r2adj)

if equation.Fstat == None:
    print('Model F-statistic: n/a')
else:
    print('Model F-statistic:',  equation.Fstat)

if equation.Fpv == None:
    print('Model F-statistic p-value: n/a')
else:
    print('Model F-statistic p-value:',  equation.Fpv)

if equation.ll == None:
    print('Model log-likelihood: n/a')
else:
    print('Model log-likelihood:',  equation.ll)

if equation.aic == None:
    print('Model AIC: n/a')
else:
    print('Model AIC:',  equation.aic)

if equation.bic == None:
    print('Model BIC: n/a')
else:
    print('Model BIC:',  equation.bic)


print()
print("Individual Parameter Statistics:")
for i in range(len(equation.solvedCoefficients)):
    if type(equation.tstat_beta) == type(None):
        tstat = 'n/a'
    else:
        tstat = '%-.5E' %  ( equation.tstat_beta[i])

    if type(equation.pstat_beta) == type(None):
        pstat = 'n/a'
    else:
        pstat = '%-.5E' %  ( equation.pstat_beta[i])

    if type(equation.sd_beta) != type(None):
        print("Coefficient %s = %-.16E, std error: %-.5E" % (equation.GetCoefficientDesignators()[i], equation.solvedCoefficients[i], equation.sd_beta[i]))
    else:
        print("Coefficient %s = %-.16E, std error: n/a" % (equation.GetCoefficientDesignators()[i], equation.solvedCoefficients[i]))
    print("          t-stat: %s, p-stat: %s, 95 percent confidence intervals: [%-.5E, %-.5E]" % (tstat,  pstat, equation.ci[i][0], equation.ci[i][1]))


print()
print("Coefficient Covariance Matrix:")
for i in  equation.cov_beta:
    print(i)

print()
##########################################################
# graphics output section
def ModelScatterConfidenceGraph(equation, graphWidth, graphHeight):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    axes = f.add_subplot(111)
    y_data = equation.dataCache.allDataCacheDictionary['DependentData']
    x_data = equation.dataCache.allDataCacheDictionary['IndependentData'][0]

    # first the raw data as a scatter plot
    axes.plot(x_data, y_data,  'D')

    # create data for the fitted equation plot
    xModel = numpy.linspace(min(x_data), max(x_data))

    tempcache = equation.dataCache # store the data cache
    equation.dataCache = pyeq3.dataCache()
    equation.dataCache.allDataCacheDictionary['IndependentData'] = numpy.array([xModel, xModel])
    equation.dataCache.FindOrCreateAllDataCache(equation)
    yModel = equation.CalculateModelPredictions(equation.solvedCoefficients, equation.dataCache.allDataCacheDictionary)
    equation.dataCache = tempcache # restore the original data cache

    # now the model as a line plot
    axes.plot(xModel, yModel)

    # now calculate confidence intervals
    # http://support.sas.com/documentation/cdl/en/statug/63347/HTML/default/viewer.htm#statug_nlin_sect026.htm
    # http://www.staff.ncl.ac.uk/tom.holderness/software/pythonlinearfit
    mean_x = numpy.mean(x_data)
    n = equation.nobs

    t_value = scipy.stats.t.ppf(0.975, equation.df_e) # (1.0 - (a/2)) is used for two-sided t-test critical value, here a = 0.05

    confs = t_value * numpy.sqrt((equation.sumOfSquaredErrors/equation.df_e)*(1.0/n + (numpy.power((xModel-mean_x),2.0)/
                                                                                       ((numpy.sum(numpy.power(x_data,2.0)))-n*(numpy.power(mean_x,2.0))))))

    # get lower and upper confidence limits based on predicted y and confidence intervals
    upper = yModel + abs(confs)
    lower = yModel - abs(confs)

    # mask off any numbers outside the existing plot limits
    booleanMask = yModel > axes.get_ylim()[0]
    booleanMask &= (yModel < axes.get_ylim()[1])

    # color scheme improves visibility on black background lines or points
    axes.plot(xModel[booleanMask], lower[booleanMask], linestyle='solid', color='white')
    axes.plot(xModel[booleanMask], upper[booleanMask], linestyle='solid', color='white')
    axes.plot(xModel[booleanMask], lower[booleanMask], linestyle='dashed', color='blue')
    axes.plot(xModel[booleanMask], upper[booleanMask], linestyle='dashed', color='blue')

    # here you can change lables for axes
    axes.set_title('') # add a title
    axes.set_xlabel('X') # X axis data label
    axes.set_ylabel('Y') # Y axis data label

    plt.show()
    plt.close('all') # clean up after using pyplot


graphWidth = 700
graphHeight = 500
ModelScatterConfidenceGraph(equation, graphWidth, graphHeight)