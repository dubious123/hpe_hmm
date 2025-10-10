import os
import sys
import time
import numpy as np
import pickle
from pathlib import Path

from tabulate import tabulate
from dex_ycb_toolkit.factory import get_dataset
from collections import defaultdict

from dex_ycb_toolkit.hpe_eval import HPEEvaluator

from sklearn.preprocessing import StandardScaler

# from factory import get_dataset
# from hpe_logging import get_logger

from hmmlearn.hmm import GaussianHMM
from joblib import dump, load

def generate_anno_file(name):
    '''
    Args:
      name: Dataset name. E.g., 's0_test'.
    '''

    print("Generating HPE annotation file")
    s = time.time()
    
    dataset = get_dataset(name)

    joint_3d_dict = defaultdict(dict)

    for i in range(len(dataset)):
        if (i + 1) in np.floor(np.linspace(0, len(dataset), 11))[1:]:
            print(
                "{:3.0f}%  {:6d}/{:6d}".format(
                    100 * i / len(dataset), i, len(dataset)
                )
            )

        
        sample = dataset[i] 
        label_file = sample["label_file"]
        key = "\\".join(label_file.split("dex_ycb\\", 1)[-1].split("\\")[:-1])
        label = np.load(label_file)
        joint_3d = label["joint_3d"].reshape(21,3)

        if np.all(joint_3d == -1):
            continue

        joint_3d *= 1000
        joint_3d_dict[key][i] = joint_3d

    print("# total samples: {:6d}".format(len(dataset)))
    print("# valid samples: {:6d}".format(sum(len(frame_dict) for frame_dict in joint_3d_dict.values())))
    anno = {"joint_3d" : joint_3d_dict}

    with open(os.path.join(os.path.dirname(__file__), "results", f"anno_hpe_hmm_{name}.pkl"), "wb") as f:
        pickle.dump(anno, f)

    e = time.time()
    print("time: {:7.2f}".format(e - s))

def load_anno_file(name):
    anno_file = os.path.join(os.path.dirname(__file__), "results", f"anno_hpe_hmm_{name}.pkl")
    
    if os.path.isfile(anno_file):
        print("Found HPE annotation file for hmm.")
    else:
        print("Cannot find HPE annotation file for hmm.")
        generate_anno_file(name)
    
    print("loading anno_hmm_file")
    with open(anno_file, "rb") as f:
        anno = pickle.load(f)
    print("done")

    for seq_key, frame_dict in anno["joint_3d"].items():
        anno["joint_3d"][seq_key] = {k: v.astype(np.float64) for k, v in frame_dict.items()}
    return anno["joint_3d"]

def load_results(res_file, joint_3d_gt):
    results = defaultdict(dict)

    frame_to_seq = {}
    for seq_key, frame_dict in joint_3d_gt.items():
        for frame_id in frame_dict.keys():
            frame_to_seq[frame_id] = seq_key

    with open(res_file, "r") as f:
        for line in f:
            elems = line.split(",")
            if len(elems) != 64:
                raise ValueError(
                    "a line does not have 64 comma-seperated elements: {}".format(
                        line
                    )
                )
            frame_id = int(elems[0])
            joint_3d = np.array(elems[1:], dtype=np.float64).reshape(21, 3)

            seq_key = frame_to_seq.get(frame_id)
            if seq_key is None:
                print(f"[WARN] frame_id {frame_id} not found in GT")
                continue

            results[seq_key][frame_id] = joint_3d
    return results

def save_results(hmm_result, out_file):
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for i, row_data in enumerate(hmm_result):
            row = [str(i)] + ["{:.13f}".format(x) for x in row_data]
            f.write(",".join(row) + "\n")
    return

def hmm_train(name, model_file_name, override = True):
    model_path = os.path.join(os.path.dirname(__file__), "results", f"{model_file_name}.joblib")
    if override is False and os.path.isfile(model_path):
        return

    joint_3d_gt = load_anno_file(name)

    model = GaussianHMM(
       n_components = 5,
       covariance_type = "diag",
       #covariance_type = "tied",
       n_iter = 100,
       random_state=19990827,
       verbose=True)

    all_X, lengths  = [], []

    for seq_id, frame_dict in joint_3d_gt.items():
        frame_ids = sorted(frame_dict.keys())
        X = np.stack([frame_dict[fid].reshape(-1) for fid in frame_ids], axis=0)
        all_X.append(X)
        lengths.append(len(X))

    X_all = np.concatenate(all_X, 0)
    
    scaler = StandardScaler().fit(X_all)
    Z_all = scaler.transform(X_all)

    model.fit(Z_all, lengths)

    dump({"model": model, "scaler": scaler}, model_path)
    print(f"model saved to: {model_path}")

def hmm_val_temp(name, model_file_name, out_file):
    model_path = os.path.join(os.path.dirname(__file__), "results", f"{model_file_name}.joblib")
    bundle = load(model_path)
    model, scaler = bundle["model"], bundle["scaler"]

    joint_3d_gt = load_anno_file(name)
    preds = joint_3d_gt
    results = defaultdict(dict)

    for seq_id, frames in preds.items():
        frame_ids = sorted(frames.keys())
        X = np.stack([frames[fid].reshape(-1) for fid in frame_ids], axis=0) # (T, 63)

        Z = scaler.transform(X) if scaler is not None else X

        gamma = model.predict_proba(Z)

        X_smooth_z = gamma @ model.means_

        X_smooth = scaler.inverse_transform(X_smooth_z) if scaler is not None else X_smooth_z

        for i, fid in enumerate(frame_ids):
            results[fid] = X_smooth[i]

    with open(out_file, "w", newline="") as f:
        for fid in sorted(results.keys()):
            vec = np.asarray(results[fid], dtype=np.float64).reshape(-1)
            if vec.size != 63:
                raise ValueError(f"frame_id {fid}: expected 63 values, got {vec.size}")
            f.write(f"{int(fid)}," + ",".join(f"{x:.4f}" for x in vec.tolist()) + "\n")

    print("out_file done")
    hpe_eval = HPEEvaluator(name)

    hpe_eval.evaluate(out_file, os.path.join(os.path.dirname(__file__), "results") )

def hmm_eval(name, model_file_name, res_file, out_file):
    model_path = os.path.join(os.path.dirname(__file__), "results", f"{model_file_name}.joblib")
    bundle = load(model_path)
    model, scaler = bundle["model"], bundle["scaler"]

    joint_3d_gt = load_anno_file(name)
    preds = load_results(res_file , joint_3d_gt)
    results = defaultdict(dict)

    for seq_id, frames in preds.items():
        frame_ids = sorted(frames.keys())
        X = np.stack([frames[fid].reshape(-1) for fid in frame_ids], axis=0) # (T, 63)

        Z = scaler.transform(X) if scaler is not None else X

        gamma = model.predict_proba(Z)

        X_smooth_z = gamma @ model.means_

        X_smooth = scaler.inverse_transform(X_smooth_z) if scaler is not None else X_smooth_z

        for i, fid in enumerate(frame_ids):
            results[fid] = X_smooth[i]

    with open(out_file, "w", newline="") as f:
        for fid in sorted(results.keys()):
            vec = np.asarray(results[fid], dtype=np.float64).reshape(-1)
            if vec.size != 63:
                raise ValueError(f"frame_id {fid}: expected 63 values, got {vec.size}")
            f.write(f"{int(fid)}," + ",".join(f"{x:.4f}" for x in vec.tolist()) + "\n")

    print("out_file done")
    hpe_eval = HPEEvaluator(name)

    hpe_eval.evaluate(out_file, os.path.join(os.path.dirname(__file__), "results") )

