import numpy as np
from MDLP import MDLP_Discretizer
import pandas as pd
import sklearn as sp
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, Normalizer


def Discretiza(df, columna_target=0, verbose=False):
    '''
    Funcion para discretizar un dataset

    Parameters:
    ----------
    :param df:
    :param columna_target:
    :param verbose:
    :return:
    '''

    # Preparamos los datos
    columna_target_end = columna_target + 1
    X = df.iloc[:, 0:len(df)].values
    y = np.transpose(df.iloc[:, columna_target:columna_target_end].values)[0]
    numeric_features = np.arange(X.shape[1])  # all features in this dataset are numeric. These will be discretized


    # Initialize discretizer object and fit to training data
    discretizer = MDLP_Discretizer(features=numeric_features)
    discretizer.fit(X, y)
    X_discretized = discretizer.transform(X)


    # Print a slice of original and discretized data
    if verbose:
        print('Original dataset:\n%s' % str(X[0:5]))
        print('Discretized dataset:\n%s' % str(X_discretized[0:5]))

        print('see how feature 0 was discretized')
        print('Interval cut-points: %s' % str(discretizer._cuts[0]))
        print('Bin descriptions: %s' % str(discretizer._bin_descriptions[0]))

    return X, X_discretized


def Estandariza(datos_normalizados, method=0):
    if method == 0:
        df = pd.DataFrame(sp.preprocessing.normalize(datos_normalizados))
    elif method == 1:
        normalizador = Normalizer().fit(datos_normalizados)
        df = pd.DataFrame(normalizador.transform(datos_normalizados))
    elif method == 2:
        MinMaxNormalizador = MinMaxScaler(feature_range=(0, 1))
        df = pd.DataFrame(MinMaxNormalizador.fit_transform(datos_normalizados))
    elif method == 3:
        df = pd.DataFrame(zscore(datos_normalizados))

    return df


def Computa_Discretizacion(target=0, method=0, stats=False , explore_methods = False):
    '''
    Function to test that the two files are working correctly
    :param target:
    :param method:
    :param stats:
    :param explore_methods:
    :return:
    '''

    ######### USE-CASE EXAMPLE #############

    # Read the file
    data = pd.read_csv("./Data/Datos_discretizar.csv")

    # check that all of the are numbers
    data.dtypes()
    # create the casting
    cast_float = lambda x: pd.to_numeric(x.str.replace(',', '.'), downcast="float")
    data.applymap(cast_float)

    # Standarize
    data_stand = sp.preprocessing.scale(data.values)

    if explore_methods:
        solution = pd.DataFrame()
        solution['dato_original_'+list(datos)[target]] = datos['Avg_position'][0:len(datos_normalizados[:, target])]
        solution['dato_normalizado'] = pd.DataFrame(datos_normalizados)[0][0:len(datos_normalizados[:, target])]
        for i in range(4):
            df = Estandariza(datos_normalizados,i)
            X, X_discretized = Discretiza(df)
            solution['dato_estandarizado_salida_metodo_'+str(i)] = X[:, target]
            solution['dato_discretizado_'+str(i)] = X_discretized[:, target]
        solution.to_csv("./Resultados/Prueba_Algoritmo_discretizador_con_metodos_estandarizacion_v2_"+list(datos)[target]+".csv",
                        sep=";",
                        index=False,
                        encoding="utf-8")

    else:
        # Estandarizamos
        if method == 0:
            df = pd.DataFrame(sp.preprocessing.normalize(datos_normalizados))
        elif method == 1:
            normalizador = Normalizer().fit(datos_normalizados)
            df = pd.DataFrame(normalizador.transform(datos_normalizados))
        elif method == 2:
            MinMaxNormalizador = MinMaxScaler(feature_range=(0, 1))
            df = pd.DataFrame(MinMaxNormalizador.fit_transform(datos_normalizados))
        elif method == 3:
            df = pd.DataFrame(zscore(datos_normalizados))

        if stats:
            print( df.describe())

        # Discretizamos
        X, X_discretized = Discretiza(df)

        solution = pd.DataFrame()
        solution['dato_original_'+list(datos)[target]] = datos['Avg_position'][0:len(X[:, target])]
        solution['dato_normalizado'] = pd.DataFrame(datos_normalizados)[0][0:len(X[:, target])]
        solution['dato_normalizado_salida'] = X[:, target]
        solution['dato_discretizado'] = X_discretized[:, target]

        solution.to_csv("./Resultadosx/Prueba_Algoritmo_discretizador"+list(datos)[target]+".csv", sep=";" , index=False, encoding="utf-8")

if __name__ == '__main__':
    Computa_Discretizacion(explore_methods=True)