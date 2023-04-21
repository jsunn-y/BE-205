import yaml
import pickle
import numpy as np
from tqdm.auto import tqdm

def run_test_case(X, y):

    n_markers = X.shape[3]

    whole_cell_y = y[:, :, :, 0].squeeze()
    n_cells = np.max(whole_cell_y) + 1

    # Normalize image # e.g., adaptive histogram normalization
    for channel in range(n_markers):
        img = X[:, :, channel]
        #norm_img = (img - np.min(img)) / (np.max(img) - np.min(img))
        norm_img = (img - np.mean(img)) / np.std(img)
        X[:,:, channel] = norm_img

    # Initialize a marker expression panel with shape=(number of celltypes, number of markers)
    means_X_bycell = np.zeros((n_cells, n_markers))
        
    with tqdm() as pbar:
        pbar.reset(n_cells)

        # Iterate through all the cell views
        for i in range(n_cells):
            # Collect views of each cell and associated celltype
            view = np.where(whole_cell_y == i)
            view_X = X[:, view[0], view[1], :]

            # Take the mean marker intensity for each marker
            means_X = np.mean(view_X, axis = 1)
            means_X_bycell[i, :] = means_X
            pbar.update()

    clf = pickle.load(open('model.sav', 'rb'))
    y_pred = clf.predict(means_X_bycell)
    keys = np.arange(n_cells)

    return dict(zip(keys, y_pred))