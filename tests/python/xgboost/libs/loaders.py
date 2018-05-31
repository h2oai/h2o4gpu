import os
import pandas as pd
import arff
import numpy as np
from functools import reduce
import sqlite3
import logging
from libs.planet_kaggle import to_multi_label_dict, get_file_count, enrich_with_feature_encoding, featurise_images, generate_validation_files
import tensorflow as tf
from keras.applications.resnet50 import ResNet50


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_FRAUD_PATH = 'fraud_detection', 'credit_card_fraud_kaggle', 'creditcard.csv'
_IOT_PATH = 'iot', 'sensor_stream_berkeley', 'sensor.arff'
_AIRLINE_PATH = 'airline', 'airline_14col.data'
_FOOTBALL_PATH = 'football', 'database.sqlite'
_BCI_PATH = 'bci', 'data.npz'
_HIGGS_PATH = 'higgs', 'HIGGS.csv'
_KAGGLE_ROOT = 'planet'
_PLANET_KAGGLE_LABEL_CSV = 'train_v2.csv'
_PLANET_KAGGLE_TRAIN_DIR = 'train-jpg'
_PLANET_KAGGLE_VAL_DIR = 'validate-jpg'


def _get_datapath():
    try:
        datapath = os.environ['MOUNT_POINT']
    except KeyError:
        logger.info("MOUNT_POINT not found in environment. Defaulting to /fileshare")
        datapath = '/fileshare'
    return datapath


def load_fraud():
    """ Loads the credit card fraud data

    The datasets contains transactions made by credit cards in September 2013 by european cardholders.
    This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
    The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
    It contains only numerical input variables which are the result of a PCA transformation.

    Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about
    the data.
    Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed
    with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first
    transaction in the dataset.
    The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. 
    Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
    Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve
    (AUPRC).
    Confusion matrix accuracy is not meaningful for unbalanced classification.

    The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group
    (http://mlg.ulb.ac.be) of ULB (Universite Libre de Bruxelles) on big data mining and fraud detection. More details 
    on current  and past projects on related topics are available on http://mlg.ulb.ac.be/BruFence 
    and http://mlg.ulb.ac.be/ARTML
    Please cite: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with
    Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

    Returns
    -------
    pandas DataFrame

    """
    return pd.read_csv(reduce(os.path.join, _FRAUD_PATH, _get_datapath()))


def load_iot():
    """ Loads iot data

    Sensor stream contains information (temperature, humidity, light, and sensor voltage) collected from 54 sensors deployed
    in Intel Berkeley Research Lab. The whole stream contains consecutive information recorded over a 2 months
    period (1 reading per 1-3 minutes). I used the sensor ID as the class label, so the learning task of the stream is
    to correctly identify the sensor ID (1 out of 54 sensors) purely based on the sensor data and the corresponding recording
    time.

    While the data stream flow over time, so does the concepts underlying the stream. For example, the lighting during
    the working hours is generally stronger than the night, and the temperature of specific sensors (conference room)
    may regularly rise during the meetings.

    Returns
    -------
    pandas DataFrame
    """
    dataset = arff.load(open(reduce(os.path.join, _IOT_PATH, _get_datapath())))
    columns = [i[0] for i in dataset['attributes']]
    return pd.DataFrame(dataset['data'], columns=columns)


def load_airline():
    """ Loads airline data
    The dataset consists of a large amount of records, containing flight arrival and departure details for all the 
    commercial flights within the USA, from October 1987 to April 2008. Its size is around 116 million records and 
    5.76 GB of memory.
    There are 13 attributes, each represented in a separate column: Year (1987-2008), Month (1-12), Day of Month (1-31), 
    Day of Week (1:Monday - 7:Sunday), CRS Departure Time (local time as hhmm), CRS Arrival Time (local time as hhmm), 
    Unique Carrier, Flight Number, Actual Elapsed Time (in min), Origin, Destination, Distance (in miles), and Diverted
    (1=yes, 0=no). 
    The target attribute is Arrival Delay, it is a positive or negative value measured in minutes. 
    Link to the source: http://kt.ijs.si/elena_ikonomovska/data.html

    Returns
    -------
    pandas DataFrame
    """
    cols = ['Year', 'Month', 'DayofMonth', 'DayofWeek', 'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'ActualElapsedTime', 'Origin', 'Dest', 'Distance', 'Diverted', 'ArrDelay']
    return pd.read_csv(reduce(os.path.join, _AIRLINE_PATH, _get_datapath()), names=cols)


def load_football():
    """ Loads football data
    Dataset of football stats. +25,000 matches, +10,000 players from 11 European Countries with their lead championship
    Seasons 2008 to 2016. It also contains players attributes sourced from EA Sports' FIFA video game series,
    including the weekly updates, team line up with squad formation (X, Y coordinates), betting odds from up to 10 
    providers and detailed match events (goal types, possession, corner, cross, fouls, cards etc...) for +10,000 matches.
    The meaning of the columns can be found here: http://www.football-data.co.uk/notes.txt
    Number of attributes in each table (size of the dataframe):
    countries (11, 2)
    matches (25979, 115)
    leagues (11, 3)
    teams (299, 5)
    players (183978, 42)
    Link to the source: https://www.kaggle.com/hugomathien/soccer
    
    Returns
    -------
    list of pandas DataFrame
    """
    database_path = reduce(os.path.join, _FOOTBALL_PATH, _get_datapath())
    with sqlite3.connect(database_path) as con:
        countries = pd.read_sql_query("SELECT * from Country", con)
        matches = pd.read_sql_query("SELECT * from Match", con)
        leagues = pd.read_sql_query("SELECT * from League", con)
        teams = pd.read_sql_query("SELECT * from Team", con)
        players = pd.read_sql("SELECT * FROM Player_Attributes;", con)
    return countries, matches, leagues, teams, players


def load_bci():
    """ Loads BCI data

    Contains measurements from 64 EEG sensors on the scalp of a single participant. 
    The purpose of the recording is to determine from the electrical brain activity when the participant is paying attention.

    Returns
    -------
    A tuple containing four numpy arrays
        train features
        train labels
        test features
        test labels
    """

    npzfile = np.load(reduce(os.path.join, _BCI_PATH, _get_datapath()))
    return npzfile['train_X'], npzfile['train_y'], npzfile['test_X'], npzfile['test_y']



def load_higgs():
    """ Loads HIGGS data
    
    Dataset of atomic particles measurements. The total size of the data is 11 millions of observations. 
    It can be used in a classification problem to distinguish between a signal process which produces Higgs 
    bosons and a background process which does not.
    The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22) are kinematic 
    properties measured by the particle detectors in the accelerator. The last seven features are functions of 
    the first 21 features; these are high-level features derived by physicists to help discriminate between the 
    two classes. The first column is the class label (1 for signal, 0 for background), followed by the 28 
    features (21 low-level features then 7 high-level features): lepton pT, lepton eta, lepton phi, 
    missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, 
    jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, 
    jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb.
    Link to the source: https://archive.ics.uci.edu/ml/datasets/HIGGS
    
    Returns
    -------
    pandas DataFrame
    """
    cols = ['boson','lepton_pT','lepton_eta','lepton_phi','missing_energy_magnitude','missing_energy_phi','jet_1_pt','jet_1_eta','jet_1_phi','jet_1_b-tag','jet_2_pt','jet_2_eta','jet_2_phi','jet_2_b-tag','jet_3_pt','jet_3_eta','jet_3_phi','jet_3_b-tag','jet_4_pt','jet_4_eta','jet_4_phi','jet_4_b-tag','m_jj','m_jjj','m_lv','m_jlv','m_bb','m_wbb','m_wwbb']
    return pd.read_csv(reduce(os.path.join, _HIGGS_PATH, _get_datapath()), names=cols)


def load_planet_kaggle():
    """ Loads Planet Kaggle data
    
    Dataset of satellite images of the Amazon. The objective of this dataset is to label satellite image chips 
    with atmospheric conditions and various classes of land cover/land use. Resulting algorithms will help the 
    global community better understand where, how, and why deforestation happens all over the world. The images
    use the GeoTiff format and each contain four bands of data: red, green, blue, and near infrared.
    To treat the images we used transfer learning with the CNN ResNet50. The images are featurized with this
    deep neural network. Once the features are generated we can use a boosted tree to classify them.
    Link to the source: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
    
    Returns
    -------
    A tuple containing four numpy arrays
        train_features
        y_train
        validation_features
        y_val
    """    
    csv_path = reduce(os.path.join, (_KAGGLE_ROOT, _PLANET_KAGGLE_LABEL_CSV), _get_datapath())
    train_path = reduce(os.path.join, (_KAGGLE_ROOT, _PLANET_KAGGLE_TRAIN_DIR), _get_datapath())
    val_path = reduce(os.path.join, (_KAGGLE_ROOT, _PLANET_KAGGLE_VAL_DIR), _get_datapath())
    assert os.path.isfile(csv_path)
    assert os.path.exists(train_path)
    if not os.path.exists(val_path): os.mkdir(val_path)
    if not os.listdir(val_path): 
        logger.info('Validation folder is empty, moving files...')
        generate_validation_files(train_path, val_path)
    
    logger.info('Reading in labels')
    labels_df = pd.read_csv(csv_path).pipe(enrich_with_feature_encoding)
    multi_label_dict = to_multi_label_dict(labels_df)
    
    nb_train_samples = get_file_count(os.path.join(train_path, '*.jpg'))
    nb_validation_samples = get_file_count(os.path.join(val_path, '*.jpg'))

    logger.debug('Number of training files {}'.format(nb_train_samples))
    logger.debug('Number of validation files {}'.format(nb_validation_samples))
    logger.debug('Loading model')

    model = ResNet50(include_top=False)
    train_features, train_names = featurise_images(model, 
                                                train_path, 
                                                'train_{}', 
                                                    range(nb_train_samples),
                                                    desc='Featurising training images')

    validation_features, validation_names = featurise_images(model, 
                                                val_path, 
                                                #'test_{}',
                                                'train_{}',
                                                #range(nb_validation_samples),
                                                range(nb_train_samples, nb_train_samples+nb_validation_samples),
                                                desc='Featurising validation images')

    # Prepare data
    y_train = np.array([multi_label_dict[name] for name in train_names])
    y_val = np.array([multi_label_dict[name] for name in validation_names])

    return train_features, y_train, validation_features, y_val
