import os
import numpy as np

def clear_2frame_val_predictions(predictions, labels):
    directory = os.path.join("output","predictions")
    if not os.path.exists(directory): os.makedirs(directory)
    for i in range(len(predictions)):
        act_pred = np.fromfile(predictions[i], dtype=np.uint32).reshape((-1, 1))
        act_label = np.fromfile(labels[i], dtype=np.uint32).reshape((-1, 1))
        prev_label = np.fromfile(labels[i-1], dtype=np.uint32).reshape((-1, 1))
        if len(act_pred) == len(act_label): act_pred.tofile(os.path.join(directory, str(i).zfill(6)+".label"))
        else: act_pred[len(prev_label):].tofile(os.path.join(directory, str(i).zfill(6)+".label"))

def file_paths(directory):
    paths = []
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            paths.append(os.path.join(dirpath, f))
    return paths

pred_paths = file_paths(os.path.join("input","predictions"))
val_label_paths = file_paths(os.path.join("input","labels"))
clear_2frame_val_predictions(pred_paths, val_label_paths)