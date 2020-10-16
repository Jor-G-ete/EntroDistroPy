import pandas as pd
import yaml, os, pickle
# import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sp
import Bin_Discretizadores as Bin_dis
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from Entropy_Discretizadores import MDLP
from datetime import datetime
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from yellowbrick.classifier import ClassificationReport
from joblib import *

"This .py discretizes by entropy one variable objective, then it disctrezies the rest by Kbins"


def entro_discretizer(df, column_target="", discretizer_data={}, verbose=False):
    '''
    Function to discretize by entropy one variable of pandas at a time
    :param df: Pandas dataframe
    :param column_target: String -> name of the column to be discretized
    :param discretizer_data: Dict -> Where the cut points and binning data of each iteration will be saved
    :param verbose: Bool -> make the program verbose
    :return: Nothing
    '''

    # Extract the column asked and check that it's contained
    if column_target not in list(df): raise ValueError("The column doesn't exist in the dataframe")

    # remove cols already discretizied
    to_remove = [i for i in list(df) if "entrodistro" in i]
    df_rem = df.drop(to_remove, axis=1)

    # Select only numeric types cols
    criteria = df_rem.dtypes != 'object'

    # pass to a numpy array and drop the target column
    # df_sel = df[criteria.index[criteria]].drop(column_target, axis=1).to_numpy()
    df_sel = df_rem[criteria.index[criteria]].to_numpy()

    # save the cols left outside
    df_sel_out = df_rem[criteria.index[~criteria]].to_numpy()

    # Extract X
    X = df_sel

    # Extract Y
    y = np.transpose(df[[column_target]].to_numpy())[0]
    numeric_features = np.arange(X.shape[1])  # all features in this dataset are numeric. These will be discretized

    # Initialize discretizer object and fit to training data
    discretizer = MDLP.MDLP_Discretizer(features=numeric_features)
    discretizer.fit(X, y)
    X_discretized = discretizer.transform(X)

    # Print a slice of original and discretized data
    if verbose:
        print('Original dataset:\n%s' % str(X[0:5]))
        print('Discretized dataset:\n%s' % str(X_discretized[0:5]))

        print('see how feature 0 was discretized')
        print('Interval cut-points: %s' % str(discretizer._cuts[list(df).index(column_target)]))
        print('Bin descriptions: %s' % str(discretizer._bin_descriptions))

    # Save the new computed data in the dataframe passed and a dict
    discretizer_data[column_target+'_cuts'] = discretizer._cuts[list(df).index(column_target)]
    discretizer_data[column_target+'_bin_descriptions'] = discretizer._bin_descriptions
    df[column_target+'_entrodistro'] = X_discretized[:, list(df).index(column_target)]
    return df, discretizer_data


def normalize(df, columns=[], minmax=True, newcols=False):
    """
    Function to normalize some columns of a dataframe
    :param df: pandas dataframe
    :param columns: Array of String -> Columns to be normalized
    :param minmax: Bool -> algorithm to be used for computing the columns
    :param newcols: Bool -> Value for choosing what to return
    :return: Dataframe -> pandas dataframe with its data normalized if newcol is False,
    otherwise it will return a dataframe with the cols
    """

    # extract the column to be used
    data = df[columns].to_numpy()

    # declare the normalizer
    if minmax: normalizer = MinMaxScaler(feature_range=(0, 1)).fit_transform
    else: normalizer = Normalizer(feature_range=(0, 1)).fit_transform

    # compute the value
    norm_res = normalizer(data)

    # return the value or copy into the old pandas
    if newcols: return pd.DataFrame(norm_res, columns=columns)
    else:
        df[columns] = pd.DataFrame(norm_res, columns=columns)
        return df


def standarize(df, columns=[], method=0, newcols=False):
    '''
    Function to standrize a dataframe in pandas
    :param df: Pandas Dataframe to be standarized
    :param columns: List of Strings -> Columns to be standarized
    :param method: Int -> Method used to standrized
    :param newcols: Bool -> Value for choosing what to return
    :return: Dataframe -> pandas dataframe with its data normalized if newcol is False,
    otherwise it will return a dataframe with the cols
    '''

    # extract the column to be used
    data = df[columns].to_numpy()

    # select the algorithm
    if method == 0: std_scaler = sp.preprocessing.scale
    elif method == 1: std_scaler = StandardScaler().fit_transform
    elif method == 2: std_scaler = zscore
    else: raise ValueError("The method chosen is not valid, please select between 0,1 and 2 methods")

    # compute the value
    std_res = std_scaler(data)

    # return the value or copy into the old pandas
    if newcols: return pd.DataFrame(std_res, columns=columns)
    else:
        df[columns] = pd.DataFrame(std_res, columns=columns)
        return df


def eval_column(df, target_column='ROI', codification="onehot",
           categories=['negativo','neutro','bajo_moderado', 'bajo', 'bajo_alto','medio_moderado', 'medio', 'medio_alto',
                       'alto_moderado', 'alto', 'muy_alto'], threshold_cat=5):
    """
    Function which values a column, in this case the use-case used is with the ROI( Return of Investment ).
    It will be created three differents arrays one for the values, another one for the category and the last one
    for the anwser ( Yest or No) .
    Categories by default ( Spanish ):
    [YES]: muy alto / alto / alto moderado / medio alto / medio / medio moderado
    [NO]: bajo alto / bajo / bajo moderado / deficiente ( Negative value ) / neutro ( When it's 0 )
    :param df: Dataframe pandas
    :param target_column: String -> name of the column to be discretized
    :param codification: String -> different types of codification for the bin algorithm
    :param categories: Array of Strings -> categories to be discretized
    :param threshold_cat: Int -> Fix the threshold for the decision of the yes or no
    :return: Dataframe of pandas with the new valued-columns
    """

    # Extract the column
    ROI = df[target_column]

    # Check that the column can be casted as a float and replace commas for points
    try:
        if df.dtypes['ROI'] not in ["float64", "int64"]:
            pd.to_numeric(ROI.str.replace(',', '.'), downcast=float)
    except Exception:
        raise ValueError("Column introduced could not be converted to float, please select another column,"
                         " check it before")

    # Divide in 10 categories and extract the cut points
    data_dis = {'bins': len(categories), 'codificacion': codification}
    ROI_dis, _, enc = Bin_dis.kbins_algo(ROI, data_dis)
    cut_points = enc.bin_edges_[0]
    # Knowing that the algorithm assigns the maximum category (10) to the maximum value in the category array
    ROI_tagged = [categories[i] for i in ROI_dis]

    # At the same time we set the answer of the question if the ROI is good or no, which will be used in the naive bayes
    ROI_answer = ["YES" if i >= threshold_cat else "NO" for i in ROI_dis]

    # Save the news columns in the passed dataframe
    df[target_column+'_'+str(len(categories))+'kbins_discretized'] = ROI_dis
    df[target_column+'_tagged_by_categories'] = ROI_tagged
    df[target_column+'_answer_bayes'] = ROI_answer

    return df


def save_report(txt, save_path_file):
    '''
    Function to write a report and save it
    :param txt: String -> to be saved in a file
    :param save_path_file: String -> path file
    :return: Nothing
    '''
    text_file = open(save_path_file, "w+")
    text_file.write(txt)
    text_file.close()


def draw_confusion_matrix(df, type, save_path="./Results/Images"):
    '''
    Function to draw and save the confusion matrix of a machine learning model
    :param df:
    :param type: String -> Type of algorithm
    :param save_path: String -> Path to save the images
    :return: Nothing
    '''
    fig, ax = plt.subplots()
    cax = ax.matshow(df)
    ax.set_title('Confusion Matrix for '+type)
    fig.colorbar(cax)
    plt.savefig(save_path+"Confusion_matrix_"+type+'_'+datetime.today().strftime('%Y%m%d'))
    plt.show()


def check_folders(to_check):
    '''
    Funtion to check the folder passed, if not it will be created
    :param to_check: String -> Full path of the folder to be checked
    :return:
    '''
    if os.path.isdir(to_check): return True
    else:
        # create the folder
        os.mkdir(to_check)
        return False


def save_model(model, path="./Saved_trained_models/", filename="default_bayes.sav", save_mode=1):
    '''
    Function for saving a machine learning model already trained, this will allow to retrieve it afterwards.
    :param model: Sklearn object -> Machine learning model to be saved
    :param path: String -> Path where to save the models
    :param filename: String -> filename of the data file
    :param save_mode: Int -> use the library pickle or joblib from sklearn depending on which one doesnt produce a mistake
    :return: Nothing
    '''
    # check that folder exist
    check_folders(path)
    # save model
    if save_mode == 1: dump(model, path+filename)
    else: pickle.dump(model, open(path+filename, 'wb'))
    # display message of info
    print("Model saved in : "+path+filename)


def load_model(path="./Saved_trained_models/", filename="", load_mode=1):
    '''
    Function for retrieving a machine learning model already trained and saved.
    :param path: String -> Path where to save the models
    :param filename: String -> filename of the data file
    :param load_mode: Int -> use the library pickle or joblib from sklearn depending on which one doesnt produce a mistake
    :return: Sklearn object-> Model saved
    '''

    # check that folder exist
    if not check_folders(path): raise ValueError("The path provided didn't exist, now a folder has been created in: "+path)

    # load the model
    if load_mode == 1: model = load(path+filename)
    else: model = pickle.load(open(path+filename, 'wb'))

    # print infomartion
    print("Model loaded.")
    return model


def classifier_evaluator(gnb, data_train, target_train, data_test, target_test):
    '''
    Function to evalute how good a classifier was trained and how it behaved against a test and validation set
    :param gnb:
    :param data_train:
    :param target_train:
    :param data_test:
    :param target_test:
    :return: Nothing
    '''

    # Instantiate the classification model and visualizer
    visualizer = ClassificationReport(gnb, classes=['Won', 'Loss'])

    visualizer.fit(data_train, target_train)  # Fit the training data to the visualizer
    visualizer.score(data_test, target_test)  # Evaluate the model on the test data
    g = visualizer.show()  # Other methods -> Draw()/show()/poof() the data


def deploy_bayes(df, target_column, method=0, save_path="./Results/", dot="99991231", load_saved_model=None):
    '''
    Function which deploys a bayesian machine learning model, saves it and shows the results
    :param df: Pandas Dataframe data to be inputed into the algorithm
    :param target_column: String -> column to be the target
    :param method: Int -> Bayesian algorithm to use
    :param save_path: String -> Save path for the trained model
    :param dot: String-> Day Of Today ( DOT )
    :param load_saved_model: String -> If introduced it will try to load a machine already trained
    :return: Nothing
    '''

    # extract target and drop the target from the data
    target = df[target_column].values.astype('float64')
    df.drop(target_column, axis=1, inplace=True)

    # Select only numeric types cols
    criteria = df.dtypes != 'object'
    df_sel = df[criteria.index[criteria]]

    # save the cols left outside
    df_sel_out = df[criteria.index[~criteria]]

    # if the cols are discretized by entropy take them
    cols = [i for i in list(df) if "entrodistro" in i]
    df_rem = df_sel[cols]

    # due to ROI is the question to be solved if it's present it's removed
    if "ROI" in [i.upper() for i in list(df_rem)]:
        df_rem.drop("ROI", axis=1, inplace=True)

    # convert to a matrix and then to a float64
    data_mx = df_rem.values.astype('float64')

    # Divide between train and test
    data_train, data_test, target_train, target_test = sp.model_selection.train_test_split(data_mx, target, test_size=0.3)

    # Divided between test and validation
    data_test, data_validation, target_test, target_validation_test = sp.model_selection.train_test_split(data_test, target_test, test_size = 0.1)

    if load_saved_model is None or load_saved_model == "":
        if method == 0:
            model = GaussianNB()
        elif method == 1:
            model = MultinomialNB()
        elif method == 2:
            model = ComplementNB()
        else:
            model = BernoulliNB()
    else: model = load_model(filename=load_saved_model)

    # train
    model.fit(data_train, target_train)

    # test
    predicted_test_fin = model.predict(data_test)
    test_report = metrics.classification_report(target_test, predicted_test_fin)
    confusion_matrix_test = pd.DataFrame(metrics.confusion_matrix(target_test, predicted_test_fin))
    # print("Predicted: "+str(predicted))
    # print("Expected: "+str(target))

    # validation
    predicted_validation_fin = model.predict(data_validation)
    validation_report = metrics.classification_report(target_validation_test, predicted_validation_fin)
    confusion_matrix_validation = pd.DataFrame(metrics.confusion_matrix(target_validation_test, predicted_validation_fin))

    # ## ---- TODO: To be improved or transformed
    # Likelihood tables
    # LikeLiHood = pd.DataFrame(model.predict_proba(data_mx), columns=list(range(2,11)))
    # LikeLiHood_log = pd.DataFrame(model.predict_log_proba(data_mx), columns=list(range(2,11)))
    # print("Possible likelihood table: "+str(model.predict_proba(data_mx)))
    # print("Possible likelihood table log: "+str(model.predict_log_proba(data_mx)))

    # Evaluate with yellowbrick the classifier
    # classifier_evaluator(model, data_mx, target, data_mx, target)
    # ## ---- END of: To be improved or transformed


    # Save the data
    save_report(test_report, save_path+'Test_report_'+type(model).__name__+'_'+dot+'.txt')
    save_report(validation_report, save_path+'Validation_report'+type(model).__name__+'_'+dot+'.txt')
    confusion_matrix_test.to_csv(save_path+'Confusion_matrix_test_'+type(model).__name__+'_'+dot+'.csv', sep=";", encoding='utf-8', index=False)
    confusion_matrix_validation.to_csv(save_path+'Confusion_matrix_validation_'+type(model).__name__+'_'+dot+'.csv', sep=";", encoding='utf-8', index=False)
    draw_confusion_matrix(confusion_matrix_test, "test"+'_'+type(model).__name__)
    draw_confusion_matrix(confusion_matrix_validation, "validation"+'_'+type(model).__name__)

    # Save the model
    save_model(model, filename=type(model).__name__+".sav")


def get_correlation(df, size, save_path="./Results/Images/"):
    '''
    Function to get the correlation matrix and save it as image
    :param df: Dataframe of pandas
    :param size: Int -> size of the picture as an array (x,y)
    :param save_path: String -> which shows the path
    :return: the correlation as a pandas object
    '''
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr)
    ax.set_title('Correlation matrix')
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(cax)
    plt.savefig(save_path+"Correlation_Matrix_"+datetime.today().strftime('%Y%m%d'))
    plt.show()
    return corr


def generate_graph(df_likelihood):
    '''
    Function to be improved but made with networkx, in which it will create a graph using the data acquiered
    :param df_likelihood: pandas datframe
    :return: Graph of networkx
    '''
    pass


def entrodistro_test(conf_file="./Entropy_dis_conf.yaml"):
    '''
    Main function which is able to disctretize by entropy among other things such as: standarize, normalize, correlation
    and deploy machine learning model of the type Naive Bayesian.
    :param conf_file: String -> path where is located the configuration file
    :return: Nothing
    '''

    # Load config file
    with open(conf_file, 'r') as stream:
        try:
            data_conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            print("Error loading the file, stoping execution")
            exit(-1)

    # load the file
    data = pd.read_csv(data_conf['input_path']+data_conf['input_file'], sep=data_conf['input_sep'], encoding="utf-8")

    # check that the path of output exist
    check_folders(data_conf['output_path'])

    # data of trial
    data = data[['Avg_cpc', 'Max_cpc', 'Net_revenue', 'Cost', 'Avg_position', 'ROI']]

    # Build a column which values in bunchs if ROI goes up or down
    data = eval_column(data)

    # normalize, take the columns(only numeric ones) from the config file to be normalize,
    # to set the range of this values between [0,1]
    data_norm = normalize(data, data_conf['columns_to_normalize'], True)

    # standarize, take the columns(only numeric ones) from the config file to be normalize,
    # to set the std of 1 and a mean of 0
    data_std = standarize(data, data_conf['columns_to_standarize'],
                                   data_conf['std_method'])

    # Apply the standarization over the normalization
    data_norm_std = standarize(data_norm, data_conf['columns_to_standarize'],
                                              data_conf['std_method'])

    # save every file
    data_norm.to_csv(data_conf['output_path']+data_conf['input_file'].replace('.csv', '')+'_' +
                                     data_conf['output_file'].replace('.csv', '') + '_normalized_'+
                                     datetime.today().strftime('%Y%m%d')+'.csv', sep=data_conf['output_sep'],
                                     encoding="utf-8", index=False, index_label=False)

    data_std.to_csv(data_conf['output_path']+data_conf['input_file'].replace('.csv', '')+'_' +
                                     data_conf['output_file'].replace('.csv', '') + '_standarized_'+
                                     datetime.today().strftime('%Y%m%d')+'.csv', sep=data_conf['output_sep'],
                                     encoding="utf-8", index=False, index_label=False)

    data_norm_std.to_csv(data_conf['output_path']+data_conf['input_file'].replace('.csv', '')+'_' +
                                     data_conf['output_file'].replace('.csv', '') + '_normlized_standarized_'+
                                     datetime.today().strftime('%Y%m%d')+'.csv', sep=data_conf['output_sep'],
                                     encoding="utf-8", index=False, index_label=False)

    # Select the data frame to be used
    if data_conf['df_type'] == "norm":
        df_entro_dis = data_norm
    elif data_conf['df_type'] == "std":
        df_entro_dis = data_std
    else:
        df_entro_dis = data_norm_std

    # declare variable to store data from the discretizier ( Cut points and binning descripition)
    discretizier_data = {}

    # Discretize/Binning by entropy each column chosen
    for idx, i in enumerate(data_conf['cols_discretize_entropy']):
        # in order to save the results of all the column which appear in the for, in the else we must put,
        # the pandas dataframe already modified returned by the algorithm
        if idx == 0: data_entropy_dis, discretizier_data = entro_discretizer(df_entro_dis, i, discretizer_data=discretizier_data)
        else:  data_entropy_dis, discretizier_data = entro_discretizer(data_entropy_dis, i, discretizer_data=discretizier_data)

    # save the data
    data_entropy_dis.to_csv(data_conf['output_path']+data_conf['input_file'].replace('.csv', '')+'_' +
                                     data_conf['output_file'].replace('.csv', '') + '_entropy_discretized_'+
                                     datetime.today().strftime('%Y%m%d')+'.csv', sep=data_conf['output_sep'],
                                     encoding="utf-8", index=False, index_label=False)
    # save dict
    stream = open(data_conf['output_path']+data_conf['input_file'].replace('.csv', '')+ '_'
                  + data_conf['output_file'].replace('.csv', '') + '_entropy_discretizer_data'
                  + datetime.today().strftime('%Y%m%d')+'.yaml', 'wb')
    # yaml.dump(discretizier_data, stream)

    # if the correlation is asked , it's displayed
    if data_conf['Correlation']:
        df_Correlation = get_correlation(data_entropy_dis, 20)

    # Train the Bayesian machine
    if data_conf['Deploy_Bayesian']:
        df_bayes = None
        # Select the data frame to be used
        if data_conf['df_bayes_type'] == "norm": df_bayes = data_norm
        elif data_conf['df_bayes_type'] == "std": df_bayes = data_std
        elif data_conf['df_bayes_type'] == "norm_std": df_bayes = data_norm_std
        else: df_bayes = data_entropy_dis
        # execute the function
        deploy_bayes(df_bayes, data_conf['Target_column'], data_conf['Bayesian_algorithm'],
                  dot=datetime.today().strftime('%Y%m%d'), load_model=data_conf['Load_pretrained_machine'])


if __name__ == "__main__":
    entrodistro_test()