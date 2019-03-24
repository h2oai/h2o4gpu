import numpy as np
import scipy
from sklearn.model_selection import train_test_split

files = [
    '../data/combined_data_1.txt',
    '../data/combined_data_2.txt',
    '../data/combined_data_3.txt',
    '../data/combined_data_4.txt',
]

if __name__ == '__main__':
    coo_row = []
    coo_col = []
    coo_val = []

    for file_name in files:
        print('processing {0}'.format(file_name))
        with open(file_name, "r") as f:
            movie = -1
            for line in f:
                if line.endswith(':\n'):
                    movie = int(line[:-2]) - 1
                    continue
                assert movie >= 0
                splitted = line.split(',')
                user = int(splitted[0])
                rating = float(splitted[1])
                coo_row.append(user)
                coo_col.append(movie)
                coo_val.append(rating)

    print('transformation...')

    coo_col = np.array(coo_col)
    user, indices = np.unique(coo_row, return_inverse=True)
    coo_val = np.array(coo_val).astype(np.float32)

    coo_matrix = scipy.sparse.coo_matrix((coo_val, (indices, coo_col)))
    print(coo_matrix.shape)

    train, test = train_test_split(
        coo_matrix.tocsc(), test_size=0.2, random_state=42)
    print(train.shape, test.shape)
    scipy.sparse.save_npz('../data/netflix_train.npz', train)
    scipy.sparse.save_npz('../data/netflix_test.npz', test)
    np.savez_compressed('../data/netflix_user.npz', user)
