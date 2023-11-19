import numpy as np
import pandas as pd


class PCA:
    """
    Michal Mackanic
    14/11/2023 v1.0

    Description
    -----------
    Class implementing principal component analysis.

    Example 1
    ---------
    data = pd.DataFrame()
    data["gdp"] = np.array([-0.0453, 0.0230, 0.0176, -0.0072, -0.0040, 0.0226, 0.0546, 0.0245, 0.0535, 0.0321, 0.0297, -0.0552, 0.0350, 0.0235], float)
    data["unpl"] = np.array([0.0710, 0.0690, 0.640, 0.0710, 0.0670, 0.0570, 0.0450, 0.0360, 0.0240, 0.0210, 0.0200, 0.0300, 0.0220, 0.0220], float)
    data["infl"] = np.array([0.0060, 0.0120, 0.0220, 0.0350, 0.0140, 0.0040, 0.0030, 0.0060, 0.0240, 0.0200, 0.0260, 0.0330, 0.0330, 0.1480], float)
    pca = PCA(data)
    reduced_data = pca.reduce(pcs_no=2)
    results = pca.results()

    Example 2
    ---------
    data = pd.read_csv("../data/eu_gov_yld_crv_shifts.csv")
    col_nms = [col_nm for col_nm in list(data.columns) if col_nm != "DATE"]
    data = data[col_nms]
    pca = PCA(data)
    reduced_data = pca.reduce(pcs_no=2)
    results = pca.results()
    """

    def __init__(self, data):
        """
        Description
        -----------
        This function implements principal component decomposition.

        Parameters
        ----------
        data : pd.DataFrame
            dataframe containing data for PCA

        Returns
        -------
        None
        """

        # store data
        self.data = {}
        self.data["original"] = data

        # get column names
        col_nms = list(self.data["original"].columns)

        # standardize data
        self.data["mean"] = {}
        self.data["stdev"] = {}
        self.data["standardized"] = pd.DataFrame()
        for col_nm in col_nms:
            self.data["mean"][col_nm] = self.data["original"][col_nm].mean()
            self.data["stdev"][col_nm] = self.data["original"][col_nm].std()
            self.data["standardized"][col_nm] = (self.data["original"][col_nm] - self.data["mean"][col_nm]) / self.data["stdev"][col_nm]

        # calculate correlation matrix
        corr_mtrx = np.matmul(self.data["standardized"].values.T,
                              self.data["standardized"].values) / (len(self.data["standardized"]) - 1)

        # decompose the correlation matrix to eigenvectors and eigenvalues
        [eigenvalues, eigenvectors] = np.linalg.eig(corr_mtrx)

        # sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1][:len(col_nms)]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # store correlation matrix, eigenvectors and eigenvalues
        self.details = {}
        self.details["corr_mtrx"] = corr_mtrx
        self.details["eigenvalues"] = eigenvalues
        self.details["eigenvectors"] = eigenvectors


    def reduce(self, pcs_no=None):
        """
        Description
        -----------
        This function drops the least important PC and re-calculate data into
        a reduced form.

        Parameters
        ----------
        pcs_no : int
            number of principal components to be taken into account

        Returns
        -------
        pd.DataFrame
            reduced data
    """

        # store column names
        col_nms = list(self.data["standardized"].columns)

        # adjust number of PCs
        if (pcs_no is None):
            pcs_no = len(col_nms)
        else:
            pcs_no = min(pcs_no, len(col_nms))

        self.details["pcs_no"] = pcs_no

        # pick up only the most important PCs
        eigenvalues_reduced = self.details["eigenvalues"][0: pcs_no]
        eigenvectors_reduced = self.details["eigenvectors"][:, 0: pcs_no]

        # calculate volatility explained
        self.details["vol_explained"] = sum(eigenvalues_reduced) / sum(self.details["eigenvalues"])

        # transform original standardized data into new co-ordinates system
        # represented by the most important PCs (if you have dropped some PCs some
        # information is lost)

        # please note that every column of data_reduced_rot has zero mean and
        # volatility equal to the corresponding eigenvalue

        data_reduced_rot =\
            np.matmul(self.data["standardized"],
                      eigenvectors_reduced)

        # transform the reduced data from new co-ordinates back to original
        # co-ordinates
        self.data["reduced"] =\
            np.matmul(eigenvectors_reduced,
                      data_reduced_rot.T).T

        # return back mean and standard deviation
        self.data["reduced"] = self.data["reduced"].set_axis(col_nms, axis=1)

        for col_nm in self.data["reduced"].columns:
            self.data["reduced"][col_nm] *= self.data["stdev"][col_nm]
            self.data["reduced"][col_nm] += self.data["mean"][col_nm]

        # return reduced data
        return self.data["reduced"]


    def results(self):
        """
        Description
        -----------
        Return details of principal component analysis, e.g. (a) eigen vectors
        and eigenvalues, (b) correlation matrix of standardized data, principal
        components considers, (c) number of considerd PCs and (d) volatility
        explained for the considered number of PCs.

        Parameters
        ----------

        Returns
        -------
        dict of objects
            corr_mtrx : np.array
                correlation matrix of standardized data
            eigenvalues : np.array
                eigenvalues
            eigenvectors : np.array
                eigenvectors
            pcs_no : int
                number of considerd principal components
            vol_explained : float
                percentage of total volatility explained through the first
                pcs_no principal components.
    """
        return self.details