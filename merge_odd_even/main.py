
# =[ Imports ]===========================================================================

import numpy as np
import os

# =[ Functions ]=========================================================================

def get_split(split_type, target_idx):
    if split_type == 'train':
        split = [0,1,2,3,4,5,6,7,9,10]
    elif split_type == 'trainval':
            split += [8]
    elif split_type == 'val':
        split = [8]
    elif split_type == 'test':
        split = [11,12,13,14,15,16,17,18,19,20,21]
    elif split_type == 'target':
        split = [target_idx]
    else:
        raise Exception('Split must be train/val/test/trainval/target')
    return split

def file_paths(directory):
    paths = []
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            paths.append(os.path.join(dirpath, f))
    return paths

def get_groundtruths_paths(gt_path, split):
    scans_paths = []
    for i_folder in split:
        scans_paths.append(
            file_paths(
                os.path.join(
                    gt_path, 
                    str(i_folder).zfill(2), 
                    "labels"
                )
            )
        )
    return scans_paths

def get_predictions_paths(pred_path, split, is_odd):
    pred_paths = []
    for i_folder in split:
        pred_paths.append(
            file_paths(
                os.path.join(
                    pred_path, 
                    str(i_folder).zfill(2), 
                    "odd_predictions" if is_odd else "even_predictions",
                    "predictions"
                )
            )
        )
    return pred_paths

def get_target_part(label_paths, pred_paths, target_idx, merge_size, is_odd):
    print("ODD - Getting target frame points...") if is_odd else print("EVEN - Getting target frame points...")
    target_preds= []
    for seq_idx in range(len(pred_paths)):
        act_target_preds = []
        for file_idx in range(len(pred_paths[seq_idx])):
            preds = np.fromfile(pred_paths[seq_idx][file_idx], dtype=np.uint32).reshape((-1, 1))
            kill_num_before = 0
            kill_num_after = len(preds)
            from_idx = 0 if file_idx - merge_size + 1 < 0 else file_idx - merge_size + 1
            to_idx = len(label_paths[seq_idx]) if file_idx + 1 > len(label_paths[seq_idx]) else file_idx + 1
            for idx, label_path in enumerate(label_paths[seq_idx][from_idx:to_idx]):
                label = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
                if idx == target_idx or len(label_paths[seq_idx][from_idx:to_idx]) == 1:
                    continue
                elif idx < target_idx:
                    kill_num_before += len(label[::2]) if is_odd else len(label[1::2])
                else:
                    kill_num_after -= len(label[::2]) if is_odd else len(label[1::2])
            act_target_preds.append(preds[kill_num_before:kill_num_after,:])
        target_preds.append(act_target_preds)
    print("Done")
    return target_preds

def reunion(arr1,arr2):
    print("Merge ODD and EVEN...")
    all_ret = []
    for s in range(len(arr1)):
        act_ret = []
        for f in range(len(arr1[s])):
            ret = np.concatenate((list(zip(arr1[s][f],arr2[s][f]))))
            if len(arr1[s][f]) != len(arr2[s][f]):
                ret = np.concatenate((ret, [arr1[s][f][-1]]))
            if f%100==0:
                print(f,"frames - done")
            act_ret.append(ret)
        all_ret.append(act_ret)
    print("Done")
    return all_ret

def save_target(target_frame_pred, split):
    print("Saving Results...")
    for i, seq in enumerate(split):
        for i, pred in enumerate(target_frame_pred[i]):
            directory = os.path.join("output", "sequences", str(seq).zfill(2), "predictions")
            if not os.path.exists(directory):
                os.makedirs(directory)
            pred.tofile(os.path.join(directory, str(i).zfill(6)+".label"))
    print("Done")

# =[ Settings ]==========================================================================

# Set target sequences
split_type = "val"  # train[0-7 & 9-10], trainval[0-10], val[8], test[11-21], target[sequence_idx]
sequence_idx = 0    # optional (if split_type = "target")

# Set the paths
gt_path = os.path.join("input", "labels", "sequences")
pred_path = os.path.join("input", "predictions", "sequences")

# Set the merge type
merge_size = 2  # How many frame was merged
target_idx = 1  # Which was the original frame

# =[ Steps ]=============================================================================

# Get the target sequences
split = get_split(split_type, sequence_idx)

# output[i][j] -> i = seq[0-split.size], j = file
# Example: If split_type="train" then gt_paths[0][3] is the 3th label path from seq 11
gt_paths = get_groundtruths_paths(gt_path, split)           
odd_pred_paths = get_predictions_paths(pred_path, split, True)
even_pred_paths = get_predictions_paths(pred_path, split, False)

# Get only the target frame's predictions
odd_target_preds = get_target_part(gt_paths, odd_pred_paths, target_idx, merge_size, True)
even_target_preds = get_target_part(gt_paths, even_pred_paths, target_idx, merge_size, False)

# Merge the odd and even part
target_frame_pred = reunion(odd_target_preds, even_target_preds)

# Save the predictions
# ToDo: Save after each merge insted of save all after all merged
save_target(target_frame_pred, split)
