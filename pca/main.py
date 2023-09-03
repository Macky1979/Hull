import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def delete_files_in_folder(folder_nm):
    """
    Michal Mackanic 03/09/2023 v1.0

    Delete all files in a folder.

    input:
        folder_nm: str
            name of folder in which all files are supposed to be deleted
    output:

    example:
        delete_files_in_folder('C:\\temp\\some_folder')
    """

    # get list of files in the folder
    file_nms = os.listdir(folder_nm)
    file_nms = [file_nm for file_nm in file_nms if os.path.isfile(folder_nm + '/' + file_nm)]

    # go file by file and try to delete it
    for file_nm in file_nms:
        file_path = os.path.join(folder_nm, file_nm)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s: %s' %(file_path, e))


def main(file_nm, pcs_no=None):
    """
    Michal Mackanic 03/09/2023 v1.0

    This function illustrates principal component analysis step by step.

    input:
        file_nm: str
            name of data file located in data folder; the first row contains
            column names and all values are numeric
        pcs_no: int
            number of PCs to be used when 'reducing' data dimension
    output:
        plots saved in folder figures
        data: pd.DataFrame
            original data as loaded from .csv file
        mean: dict
            dictionary containing original data mean per column
        stdev: dict
            dictionary containing original data standard deviation per column
        corr_mtrx: np.array
            correlation matrix based on standardized data
        eigenvalues: pd.DataFrame
            eigenvalues of the correlation matrix
        eigenvectors: pd.DataFrame
            eigenvectors of the correlation matrix
        vol_explained: float
            percentage of original volatility captured by pcs_no first PCs
        data_reduced: pd.DataFrame
            'reduced' data after applying pcs_no PCs

    example:
        file_nm = 'BankData.csv'
        pcs_no = 3
        [data, mean, stdev, corr_mtrx, eigenvalues, eigenvectors, vol_explained, data_reduced] = main(file_nm=file_nm, pcs_no=pcs_no)
    """

    # inform user
    print('Reading data from .csv file...')

    # read .csv file
    data = pd.read_csv('data//' + file_nm, low_memory=False)

    # get list of column names
    col_nms = list(data.columns)

    # inform user
    print('Standardizing data...')

    # standardize data
    mean = {}
    stdev = {}
    data_std = pd.DataFrame()
    for col_nm in col_nms:
        mean[col_nm] = data[col_nm].mean()
        stdev[col_nm] = data[col_nm].std()
        data_std[col_nm] = (data[col_nm] - mean[col_nm]) / stdev[col_nm]

    # inform user
    print('Calculating correlation matrix...')

    # calculate correlation matrix
    corr_mtrx = np.matmul(data_std.values.T, data_std.values) / (len(data_std) - 1)

    # inform user
    print('Decomposing correlation matrix into eigenvectors and eigenvalues...')

    # decompose the correlation matrix to eigenvectors and eigenvalues
    [eigenvalues, eigenvectors] = np.linalg.eig(corr_mtrx)

    # sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1][:len(col_nms)]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # inform user
    print('Picking ' + str(pcs_no) + ' the most important PCs...')

    # adjust number of PCs
    if (pcs_no is None):
        pcs_no = len(data)
    else:
        pcs_no = min(pcs_no, len(data))

    # pick up only the most important PCs
    eigenvalues_reduced = eigenvalues[0: pcs_no]
    eigenvectors_reduced = eigenvectors[:, 0: pcs_no]

    # calculate volatility explained
    vol_explained = sum(eigenvalues_reduced) / sum(eigenvalues)

    # inform user
    print('Adding row and column names to eigenvalues and eigenvectors dataframe...')

    # add column and row names to eigenvalues and eigenvectors
    pc_nms = ['pc' + str(x + 1).zfill(2) for x in range(len(col_nms))]

    eigenvalues = pd.DataFrame(eigenvalues)
    eigenvalues = eigenvalues.set_axis(pc_nms, axis=0)
    eigenvalues = eigenvalues.set_axis(['eigenvalues'], axis=1)

    eigenvectors = pd.DataFrame(eigenvectors)
    eigenvectors = eigenvectors.set_axis(col_nms, axis=0)
    eigenvectors = eigenvectors.set_axis(pc_nms, axis=1)

    # inform user
    print('Transforming original data into new co-cordinates represented by selected PCs...')

    # transform original standardized data into new co-ordinates system
    # represented by the most important PCs (if you have dropped some PCs some
    # information is lost)
    data_reduced = np.matmul(data_std, eigenvectors_reduced)

    # inform user
    print('Transforming data back into original coordinates...')

    # transform the reduced data from new co-ordinates back to original
    # co-ordinates
    data_reduced = np.matmul(eigenvectors_reduced, data_reduced.T).T

    # inform user
    print('Adding back mean and standard deviation...')

    # return back mean and standard deviation
    data_reduced = data_reduced.set_axis(col_nms, axis=1)

    for col_nm in data_reduced.columns:
        data_reduced[col_nm] *= stdev[col_nm]
        data_reduced[col_nm] += mean[col_nm]

    # inform user
    print('Deleting old figures...')

    # delete all files in folder figures
    delete_files_in_folder('figures')
    delete_files_in_folder('figures//eigenvectors')
    delete_files_in_folder('figures//orig_vs_reduced_data')

    # inform user
    print('Plot eigenvalues...')

    # plot eigenvalues
    plt.bar(pc_nms, eigenvalues.T.values[0], color ='cornflowerblue', width = 0.9)
    plt.title('Eigenvalues')
    plt.savefig('figures//eigenvalues.jpg', dpi=100)
    plt.clf()

    # plot eigenvector histograms (in ALM shock construction we assumed they
    # follow normal distribution)
    folder_nm = 'figures//eigenvectors//'
    if (len(data) > 20):

        print('Plot eigenvectors histograms...')

        for pc_nm in pc_nms:
            plt.hist(eigenvectors[pc_nm], bins=10, color='cornflowerblue', width=0.9)
            plt.title('Histogram of ' + pc_nm)
            plt.savefig(folder_nm + pc_nm + '.jpg', dpi=100)
            plt.clf()

    # inform user
    print('Plot original vs. reduced data...')

    # plot original vs. reduced data
    t = np.linspace(1, len(data), len(data))
    folder_nm = 'figures//orig_vs_reduced_data//'
    for col_nm in col_nms:
        plt.plot(t, data[col_nm].values,
                 color='red',
                 linewidth=1,
                 linestyle='-',
                 label='original data')
        plt.plot(t, data_reduced[col_nm].values,
                 color='cornflowerblue',
                 linewidth=0.75,
                 linestyle='--',
                 label='reduced data')
        plt.title(col_nm + ' - original vs. reduced data')
        plt.legend(loc='upper left')
        plt.savefig(folder_nm + col_nm + '.jpg', dpi=100)
        plt.clf()

    # inform user
    print('Plot analysis of the first two PCs...')
    plt.title('Loading factors of the first two PCs')
    for col_nm in col_nms:
        x = [0, eigenvectors.loc[col_nm, 'pc01']]
        y = [0, eigenvectors.loc[col_nm, 'pc02']]
        plt.plot(x, y, linewidth=2, label=col_nm)
    plt.legend(loc='best', fontsize="4")
    plt.xlabel('pca01')
    plt.ylabel('pca02')
    plt.savefig('figures//loading_factors.jpg', dpi=300)
    plt.clf()

    # inform user
    print('Done!')

    # return original data in original co-ordinates, means, standard deviations,
    # correlation matrix, eigenvalues, eigenvectors and "reduced " data in
    # original co-ordinates
    return data, mean, stdev, corr_mtrx, eigenvalues, eigenvectors, vol_explained, data_reduced