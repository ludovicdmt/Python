from sklearn.tree import DecisionTreeRegressor

def binningData(x, max_depth=5, min_samples_leaf=5):
    """
    Suggest good threshold to bin data.
    This function uses decision tree regression to cluster the data by using its self as the target variable.
    
    Input
    x: numerical array (Numpy)
    
    Output
    thrs_out: the recommended thresholds excluding the min and max value 
    y_hat: the average value in each bin
    
    Example:
    
    x = np.random.rand(1000)
    (thrs_out, y_hat) = binningData(x, max_depth=3, min_samples_leaf=10)
    # sort and append left- and right-most values to the bin
    thrs_out = list(np.sort(thrs_out))
    bin_outs = [min(x)] + thrs_out + [max(x)]
    # plot the histogram
    plt.hist(x=x, bins=bin_outs)
    
    kittipat@
    Jul 1, 2015
    """
    # Automatic binning
    

    # swap the data matrix
    X = np.swapaxes(np.array([x]),0,1)

    # target variable
    y = x

    # Fit regression model
    clf_1 = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    clf_1.fit(X, y)

    # Predict
    y_hat = clf_1.predict(X)

    # output bin
    thrs_out = np.unique( clf_1.tree_.threshold[clf_1.tree_.feature > -2] )
    thrs_out = np.sort(thrs_out)

    
    return (thrs_out, y_hat)
