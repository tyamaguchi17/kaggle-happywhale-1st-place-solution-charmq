import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

root = Path("./happywhale_data")


def parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train_data_dir", type=str)
    parser.add_argument("--test_data_dir", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--th", type=float, default=0.5)

    args = parser.parse_args()
    return args


def create_dataframe(num_folds, seed=0, num_records=0, phase="train"):
    if phase == "train" or phase == "valid":
        df = pd.read_csv(str(root / "train.csv"))
    elif phase == "test":
        df = pd.read_csv(str(root / "sample_submission.csv"))
        return df

    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold = -np.ones(len(df))
    for i, (_, indices) in enumerate(kfold.split(df, df["individual_id"])):
        fold[indices] = i

    df["fold"] = fold
    df.species.replace(
        {
            "globis": "short_finned_pilot_whale",
            "pilot_whale": "short_finned_pilot_whale",
            "kiler_whale": "killer_whale",
            "bottlenose_dolpin": "bottlenose_dolphin",
        },
        inplace=True,
    )
    le_species = LabelEncoder()
    le_species.classes_ = np.load(root / "species.npy", allow_pickle=True)
    le_individual_id = LabelEncoder()
    le_individual_id.classes_ = np.load(root / "individual_id.npy", allow_pickle=True)
    df["species_label"] = le_species.transform(df["species"])
    df["individual_id_label"] = le_individual_id.transform(df["individual_id"])

    if num_records:
        df = df[:num_records]

    return df


def load_embed(data_dir, train=True):
    res = np.load(data_dir)

    data = {}
    indices = res["original_index"]
    features = res["embed_features1"]

    for i, idx in enumerate(indices):
        data[idx] = features[i]

    X = []
    for i in range(len(data)):
        X.append(data[i] / np.linalg.norm(data[i]))

    X = np.array(X)
    if train:
        df = create_dataframe(5, 0, 0, "train")
        y = df["individual_id_label"].values
        return X, y
    else:
        return X


def main():
    args = parse()

    train_embeddings, train_targets = load_embed(args.train_data_dir, True)
    test_embeddings = load_embed(args.test_data_dir, False)

    neigh = NearestNeighbors(n_neighbors=100, metric="cosine")
    neigh.fit(train_embeddings)

    test_nn_distances, test_nn_idxs = neigh.kneighbors(
        test_embeddings, 100, return_distance=True
    )

    le_individual_id = LabelEncoder()
    le_individual_id.classes_ = np.load(root / "individual_id.npy", allow_pickle=True)

    th = args.th
    topk = []
    for i in tqdm(range(len(test_embeddings))):
        pred_all = le_individual_id.inverse_transform(
            train_targets[test_nn_idxs[i]]
        ).tolist()
        pred_dist = test_nn_distances[i]
        if pred_dist[0] < th:
            pred = [pred_all[0], "new_individual"]
        else:
            pred = ["new_individual", pred_all[0]]
        for i in range(len(pred_all)):
            if len(pred) == 5:
                break
            else:
                if pred_all[i] not in pred:
                    pred.append(pred_all[i])

        topk.append(pred[:5])
    topk = np.array(topk)
    df = pd.read_csv(root / "sample_submission.csv")
    pred = []
    for i in range(len(topk)):
        pred.append(" ".join(topk[i]))

    df["predictions"] = pred
    df.to_csv(args.out, index=False)
    print(df.head())
    print(
        "new individual ratio",
        (df["predictions"].map(lambda x: x.split()[0]) == "new_individual").mean(),
    )


if __name__ == "__main__":
    main()
