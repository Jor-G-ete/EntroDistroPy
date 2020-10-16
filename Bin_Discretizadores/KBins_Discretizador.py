import yaml
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
# import numpy as np


def kbins_algo(to_dis, data_conf):
    '''
    Function which creates an enconder and return the column discretized by the algorithim kbins
    :param to_dis: Series of dtype float
    :param data_conf: list of configuration
    :return: Series of dtype int, The Complete matrix of binned objects, the enconder
    '''

    # Transform the series into a numpy array
    X = to_dis.to_numpy()

    # Transform the dataset with KBinsDiscretizer
    enc = KBinsDiscretizer(n_bins=data_conf['bins'], encode=data_conf['codificacion'])
    X_binned = enc.fit_transform(X.reshape(-1, 1))
    # Fastest way
    return pd.Series(X_binned.indices), X_binned, enc
    # Manual way
    # return [np.where(x == 1)[0][0] for x in X_binned.toarray()]


def kbins_discretizator(conf_file):
    '''
    Function to discretize with the method Kbins
    :param conf_file: Path to the configuration file
    :return: nothing
    '''

    # Load configuration file
    with open(conf_file, 'r') as stream:
        try:
            data_conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            print("Error al cargar el archivo de configuracion, parando ejecucion")
            exit(-1)

    # check that the file it's a csv
    if data_conf['input_file'].split('.')[1] != "csv": raise FileExistsError('The file must be CSV not anything else')

    # Load csv
    data = pd.read_csv(data_conf['input_path']+data_conf['input_file'], sep=data_conf['input_sep'], encoding="utf-8")

    # Extract the data and remove nan if it existed due it cant be computed
    df_op = data[data_conf['columns_to_dis']].dropna()

    # Check that every column selected is a float
    # Change if a comma existed like excel or databases usually does, and change for the dot recognized by every system
    for i in data_conf['columns_to_dis']:
        df_op[i] = pd.to_numeric(df_op[i].str.replace(',', '.'), downcast="float")
        # discretize and avoid the extra results
        df_op[i+"_binned"], _, _ = kbins_algo(df_op[i], data_conf)

    # Store the data
    df_op.to_csv(data_conf['output_path']+data_conf['input_file'].split('.')[0]+'_'+data_conf['output_file'],
                       sep=data_conf['output_sep'], encoding="utf-8", index=False, index_label=False)


if __name__ == "__main__":

    # Test Zone
    conf_defecto = "./Disc_Kbins_conf.yaml"
    kbins_discretizator(conf_defecto)

