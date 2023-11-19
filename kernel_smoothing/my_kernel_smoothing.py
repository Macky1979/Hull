import numpy as np


def moving_avg_1d(x, y, k):
    """
    Michal Mackanic 12/11/2023 v1.0

    Description
    -----------
    Function applies a simple moving average on y = f(x) using values
    neighbouring with x0. Neighbourhood is defined through parameter k.

    Note: based on "Elements of Statistical Learning II", chapter 6

    Parameters
    ----------
    x : np.array of floats
        Values of idependent variable.
    y : np.array of floats
        Dependent variable; we assume some y = f(x) relation.
    k : int
        Parameter defining neighbourhood of x0 point. For k = 2 the moving
        average is defined as two points to the left and two points to the
        right from point x0, i.e. the neighbourhood is defined through 5
        points.

    Raises
    ------
    ValueError
        If weights length is different from (2 * k) + 1 or if weights do not
        sum up to 1.0.

    Returns
    -------
    np.array of floats
        Sorted values of x.
    np.array of floats
        Values of y = f(x) smoothed through a moving average.

    Example
    -------
    gdp = np.array([-0.0546, 0.0264, 0.0136, 0.0065, 0.0272, 0.0522, 0.0193, 0.0298, 0.0379, 0.0261, 0.0302, 0.0102], float)
    odr = np.array([0.0280, 0.0176, 0.0145, 0.0140, 0.0122, 0.0081, 0.0054, 0.0040, 0.0038, 0.0034, 0.0034, 0.0026], float)
    [sorted_gdp, smoothed_odr] = moving_avg_1d(x=gdp, y=odr, k=1)
    """

    # sort x and y variables
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # list holding smoothed values
    y_smoothed = []

    # go through the series and calculate moving average
    for i in range(len(x)):

        k_low = max(i - k, 0)
        k_max = min(i + k + 1, len(x) - 1)
        y_smoothed.append(np.mean(np.array(y[k_low : k_max], float)))

    # return the vector of moving averages
    return [x, np.array(y_smoothed, float)]


def epanechnikov(x, x_0, l):
    """
    Michal Mackanic 12/11/2023 v1.0

    Description
    -----------
    Epanechnikov quadratic kernel

    Note: based on "Elements of Statistical Learning II", chapter 6

    Parameters
    ----------
    x : np.array of floats
        Values of idependent variable.
    y : np.array of floats
        Dependent variable; we assume some y = f(x) relation.
    x_0 : float
        Independent variable of x0 for which the kernel should be evaluated.
    l : float
        parameter lambda od Epanechnikov quadratic kernel

    Returns
    -------
    float
        Value of kernel evaluated at point x0.

    Example
    -------
    gdp = np.array([-0.0546, 0.0264, 0.0136, 0.0065, 0.0272, 0.0522, 0.0193, 0.0298, 0.0379, 0.0261, 0.0302, 0.0102], float)
    K = epanechnikov(x=gdp[0], x_0=0.025, l=0.20)
    """

    def D(t):
        if np.abs(t) <= 1.0:
            return 3/4 * (1 - t ** 2)
        else:
            return 0.0

    t = np.abs(x - x_0) / l

    return D(t)


def kernel_smoothing_1d(x, y, x_est, l, avg_method="nadaraya_watson", kernel="epashnikov"):
    """
    Michal Mackanic 12/11/2023 v1.0

    Description
    -----------
    Smooth data using averaging method based on kernel.

    Note: based on "Elements of Statistical Learning II", chapter 6

    Parameters
    ----------
    x : np.array of floats
        Values of idependent variable.
    y : np.array of floats
        Dependent variable; we assume some y = f(x) relation.
    x_est : np.array of floats
        Values independent variables for which we should estimate y = f(x)
        using kernel smoothing.
    l : float
        Kernel parameter lambda defining "neighbourhood".
    avg_method : str, optional
        The default is "nadaraya_watson".
        Kernel-weighted average method to be applied.
    kernel : str, optional
        The default is "epashnikov".
        Kernel model to be used.

    Raises
    ------
    ValueError
        In case of unsupported combination of kernel-weighted average method
        and kernel.

    Returns
    -------
    np.array of floats
        Values of x.
    np.array of floats
        Vector of estimated values y = f(x) for x_est using kernel smoothing.

    Example 1
    ---------
    import matplotlib.pyplot as plt
    from scipy.stats import uniform
    from scipy.stats import norm

    points_no = 100
    x = uniform.rvs(loc=0, scale=1, size=points_no)
    err = norm.rvs(loc=0, scale=1/3, size=points_no)
    y = np.sin(4 * x)
    y_err = y + err

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    y_err = y_err[idx]

    [_, y_smoothed1] = moving_avg_1d(x=x, y=y_err, k=14)
    [_, y_smoothed2] = kernel_smoothing_1d(x=x, y=y_err, x_est=x, l=0.2, avg_method="nadaraya_watson", kernel="epashnikov")

    plt.title("Kernel smoothing")
    plt.xlabel("x")
    plt.ylabel("y = f(x)")
    plt.plot(x, y, "-", color="blue", label="data without noise")
    plt.plot(x, y_err, ".", color="red", label="data with noise")
    plt.plot(x, y_smoothed1, "-", color="green", label="moving average smoothed data")
    plt.plot(x, y_smoothed2, "-", color="orange", label="kernel smoothed data")
    plt.legend()

    Example 2
    ---------
    import matplotlib.pyplot as plt

    gdp = np.array([-0.0546, 0.0264, 0.0136, 0.0065, 0.0272, 0.0522, 0.0193, 0.0298, 0.0379, 0.0261, 0.0302, 0.0102], float)
    odr = np.array([0.0280, 0.0176, 0.0145, 0.0140, 0.0122, 0.0081, 0.0054, 0.0040, 0.0038, 0.0034, 0.0034, 0.0026], float)

    idx = np.argsort(gdp)
    gdp_sorted = gdp[idx]
    odr_sorted = odr[idx]

    gdp_est = np.linspace(-0.05, 0.05, 20)
    [gdp_est, odr_smoothed] = kernel_smoothing_1d(x=gdp, y=odr, x_est=gdp_est, l=0.05, avg_method="nadaraya_watson", kernel="epashnikov")

    plt.title("Kernel smoothing")
    plt.xlabel("GDP")
    plt.ylabel("ODR")
    plt.plot(gdp_sorted, odr_sorted, ".", color="red", label="original data")
    plt.plot(gdp_est, odr_smoothed, "-", color="orange", label="smoothed data")
    plt.legend()
    """

    # Nadaraya-Watson kernel-weighted average method
    def nadaraya_watson_avg(x, y, x_0, l, kernel="epanechnikov"):

        nominator = 0.0
        denominator = 0.0

        for i in range(len(x)):

            K = kernel(x_0, x[i], l)
            nominator +=  K * y[i]
            denominator += K

        return nominator / denominator

    # list holding smoothed values
    y_smoothed = []

    for i in range(len(x_est)):
        if ((avg_method == "nadaraya_watson") and (kernel == "epashnikov")):
            y_smoothed.append(nadaraya_watson_avg(x=x, y=y, x_0=x_est[i], l=l, kernel=epanechnikov))
        else:
            raise ValueError("'" + avg_method + "' and '" + kernel + "' are not supported 'average method' and 'kernel' combination!")

    # return the vector with smoothed values
    return [x_est, np.array(y_smoothed, float)]


def linreg_wght_1d(x, y, x_est, l, kernel="epashnikov"):
    """
    Michal Mackanic 12/11/2023 v1.0

    Description
    -----------
    Smooth data using weigthed linear regression method with kernel as a weight.
    Compared to kernel_smoothing_1d(), linreg_wght_1d() should perform better
    at boundary points.

    Note: based on "Elements of Statistical Learning II", chapter 6

    Parameters
    ----------
    x : np.array of floats
        Values of idependent variable.
    y : np.array of floats
        Dependent variable; we assume some y = f(x) relation.
    x_est : np.array of floats
        Values independent variables for which we should estimate y = f(x)
        using kernel smoothing.
    l : float
        Kernel parameter lambda defining "neighbourhood".
    kernel : str, optional
        The default is "epashnikov".
        Kernel model to be used.

    Raises
    ------
    ValueError
        If unsupported kernel method.

    Returns
    -------
    np.array of floats
        Values of x.
    np.array of floats
        Vector of estimated values y = f(x) for x_est using linear regression
        smoothing.

    Example 1
    ---------
    import matplotlib.pyplot as plt
    from scipy.stats import uniform
    from scipy.stats import norm

    points_no = 100
    x = uniform.rvs(loc=0, scale=1, size=points_no)
    err = norm.rvs(loc=0, scale=1/3, size=points_no)
    y = np.sin(4 * x)
    y_err = y + err

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    y_err = y_err[idx]

    [_, y_smoothed1] = moving_avg_1d(x=x, y=y_err, k=14)
    [_, y_smoothed2] = kernel_smoothing_1d(x=x, y=y_err, x_est=x, l=0.2, avg_method="nadaraya_watson", kernel="epashnikov")
    [_, y_smoothed3] = linreg_wght_1d(x=x, y=y_err, x_est=x, l=0.2, kernel="epashnikov")

    plt.title("Linear regression smoothing")
    plt.xlabel("x")
    plt.ylabel("y = f(x)")
    plt.plot(x, y, "-", color="blue", label="data without noise")
    plt.plot(x, y_err, ".", color="red", label="data with noise")
    plt.plot(x, y_smoothed1, "-", color="green", label="moving average smoothed data")
    plt.plot(x, y_smoothed2, "--", color="orange", label="kernel smoothed data")
    plt.plot(x, y_smoothed3, "-.", color="orange", label="linear regression smoothed data")
    plt.legend()

    Example 2
    ---------
    import matplotlib.pyplot as plt

    gdp = np.array([-0.0546, 0.0264, 0.0136, 0.0065, 0.0272, 0.0522, 0.0193, 0.0298, 0.0379, 0.0261, 0.0302, 0.0102], float)
    odr = np.array([0.0280, 0.0176, 0.0145, 0.0140, 0.0122, 0.0081, 0.0054, 0.0040, 0.0038, 0.0034, 0.0034, 0.0026], float)

    idx = np.argsort(gdp)
    gdp_sorted = gdp[idx]
    odr_sorted = odr[idx]

    gdp_est = np.linspace(-0.05, 0.05, 20)
    [gdp_est, odr_smoothed1] = kernel_smoothing_1d(x=gdp, y=odr, x_est=gdp_est, l=0.10, avg_method="nadaraya_watson", kernel="epashnikov")
    [gdp_est, odr_smoothed2] = linreg_wght_1d(x=gdp, y=odr, x_est=gdp_est, l=0.10, kernel="epashnikov")

    plt.title("Linear regression smoothing")
    plt.xlabel("GDP")
    plt.ylabel("ODR")
    plt.plot(gdp_sorted, odr_sorted, ".", color="red", label="original data")
    plt.plot(gdp_est, odr_smoothed1, "--", color="orange", label="kernel smoothed data")
    plt.plot(gdp_est, odr_smoothed2, "-.", color="orange", label="linear regression smoothed data")
    plt.legend()
    """

    # linear regression to estimate y = f(x) at particular point x
    def linreg_aux(x, y, x_0, l, kernel):

        # prepare inputs

        b = np.array([[1], [x_0]], float)

        B = np.ones([len(x), 2], float)
        for i in range(len(x)):
            B[i, 1] = x[i]

        W = np.zeros([len(x), len(x)], float)
        if (kernel == "epanechnikov"):
            for i in range(len(x)):
                W[i, i] = epanechnikov(x[i], x_0, l)
        else:
            raise ValueError("'" + kernel + "' kernel is not supported!")

        # perform weighted linear regression
        f = np.matmul(B.T, W)
        f = np.matmul(f, B)
        f = np.linalg.inv(f)
        f = np.matmul(b.T, f)
        f = np.matmul(f, B.T)
        f = np.matmul(f, W)
        f = np.matmul(f, y)

        # return result
        return f[0]

    # list holding smoothed values
    y_smoothed = []

    # apply smoothing through linear regression
    for i in range(len(x_est)):
        y_smoothed.append(linreg_aux(x=x, y=y, x_0=x_est[i], l=l, kernel="epanechnikov"))

    # return the vector with smoothed values
    return [x_est, np.array(y_smoothed, float)]