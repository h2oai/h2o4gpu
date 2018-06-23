import os
import numpy as np
import glob
from tqdm import tqdm
import shutil
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def labels_from(labels_df):
    """ Extracts the unique labels from the labels dataframe
    """
    # Build list with unique labels
    label_list = []
    for tag_str in labels_df.tags.values:
        labels = tag_str.split(' ')
        for label in labels:
            if label not in label_list:
                label_list.append(label)
    return label_list


def enrich_with_feature_encoding(labels_df):
    # Add onehot features for every label
    for label in labels_from(labels_df):
        labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
    return labels_df


def to_multi_label_dict(enriched_labels_df):
    df = enriched_labels_df.set_index('image_name').drop('tags', axis=1)
    return dict((filename, encoded_array) for filename, encoded_array in zip(df.index, df.values))


def get_file_count(folderpath):
    """ Returns the number of files in a folder
    """
    return len(glob.glob(folderpath))


def threshold_prediction(pred_y, threshold=0.5):# TODO: Needs to be tuned?
    return pred_y > threshold


def read_images(filepath, filenames):
    """ Read images in batches
    """
    img_data = list()
    for name in filenames:
        img_path = os.path.join(filepath, name+'.jpg')
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data.append(preprocess_input(x))
    return np.concatenate(img_data)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def featurise_images(model, filepath, nameformat, num_iter, batch_size=32, desc=None):
    """ Use DL model to featurise images
    """
    features = list()
    img_names = list()
    num_list = list(num_iter)
    num_batches = np.floor(len(num_list)/batch_size)
    
    for num_chunk in tqdm(chunks(num_list, batch_size), total=num_batches, desc=desc):
        filenames = [nameformat.format(index) for index in num_chunk]
        batch_images = read_images(filepath, filenames)
        img_names.extend(filenames)
        features.extend(model.predict_on_batch(batch_images).squeeze())
    return np.array(features), img_names
  
    
def generate_validation_files(train_path, val_path, num_train = 35000):
    """ Creates the validation files from the train files.
    """
    num_train_ini = get_file_count(os.path.join(train_path, '*.jpg'))
    assert num_train_ini > num_train
    
    order = 'mv ' + train_path + '/train_{' + str(num_train) + '..' + str(num_train_ini) + '}.jpg ' + val_path
    os.system(order)
    
    
    
