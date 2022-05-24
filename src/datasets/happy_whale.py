from pathlib import Path

import imageio
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class HappyWhaleDataset(Dataset):
    """Dataset class for happy whale."""

    ROOT_PATH = Path("./happywhale_data")

    @classmethod
    def create_dataframe(
        cls,
        num_folds: int,
        seed: int = 777,
        num_records: int = 0,
        phase: str = "train",
        pseudo_label_filename=None,
    ) -> pd.DataFrame:
        root = cls.ROOT_PATH
        if pseudo_label_filename is not None:
            if "valid" in pseudo_label_filename:
                df = pd.read_csv(
                    str(root / "pseudo_labels" / pseudo_label_filename), index_col=0
                )
                df_fb = pd.read_csv(str(root / "fullbody_train.csv")).iloc[df.index]
                df_fb_charm = pd.read_csv(str(root / "fullbody_train_charm.csv")).iloc[
                    df.index
                ]
                df_bbox_yolo = pd.read_csv(str(root / "train_bbox.csv")).iloc[df.index]
                df_bbox_detic = pd.read_csv(str(root / "train2.csv")).iloc[df.index]
                df_bbox_backfin = pd.read_csv(str(root / "train_backfin.csv")).iloc[
                    df.index
                ]
                df_bbox_backfin_charm = pd.read_csv(
                    str(root / "backfin_train_charm.csv")
                ).iloc[df.index]
            else:
                df = pd.read_csv(str(root / "pseudo_labels" / pseudo_label_filename))
                df_fb = pd.read_csv(str(root / "fullbody_test.csv"))
                df_fb_charm = pd.read_csv(str(root / "fullbody_test_charm.csv"))
                df_bbox_yolo = pd.read_csv(str(root / "test_bbox.csv"))
                df_bbox_detic = pd.read_csv(str(root / "test2.csv"))
                df_bbox_backfin = pd.read_csv(str(root / "test_backfin.csv"))
                df_bbox_backfin_charm = pd.read_csv(
                    str(root / "backfin_test_charm.csv")
                )
            df["bbox_fb"] = df_fb["bbox"].map(
                lambda x: [list(map(int, x[2:-2].split()))]
                if isinstance(x, str)
                else []
            )
            df["conf_fb"] = (
                df_fb["conf"]
                .map(lambda x: x if isinstance(x, str) else "[]")
                .map(eval)
                .map(lambda x: 0 if len(x) == 0 else x[0])
            )
            df["bbox_fb_charm"] = df_fb_charm["bbox"].map(
                lambda x: [list(map(int, x[2:-2].split()))]
                if isinstance(x, str)
                else []
            )
            df["conf_fb_charm"] = (
                df_fb_charm["conf"]
                .map(lambda x: x if isinstance(x, str) else "[]")
                .map(eval)
                .map(lambda x: 0 if len(x) == 0 else x[0])
            )

            df["bbox_yolo"] = df_bbox_yolo["bbox"].map(eval)
            df["conf_yolo"] = (
                df_bbox_yolo["conf"].map(eval).map(lambda x: 0 if len(x) == 0 else x[0])
            )

            df["bbox_detic"] = df_bbox_detic["box"].map(
                lambda x: [list(map(int, x.split()))] if isinstance(x, str) else []
            )
            df["conf_detic"] = 1

            df["bbox_backfin"] = df_bbox_backfin["bbox"].map(
                lambda x: [list(map(int, x[2:-2].split()))]
                if isinstance(x, str)
                else []
            )
            df["conf_backfin"] = (
                df_bbox_backfin["conf"]
                .map(lambda x: x if isinstance(x, str) else "[]")
                .map(eval)
                .map(lambda x: 0 if len(x) == 0 else x[0])
            )
            df["bbox_backfin_charm"] = df_bbox_backfin_charm["bbox"].map(
                lambda x: [list(map(int, x[2:-2].split()))]
                if isinstance(x, str)
                else []
            )
            df["conf_backfin_charm"] = (
                df_bbox_backfin_charm["conf"]
                .map(lambda x: x if isinstance(x, str) else "[]")
                .map(eval)
                .map(lambda x: 0 if len(x) == 0 else x[0])
            )
            le_species = LabelEncoder()
            le_species.classes_ = np.load(root / "species.npy", allow_pickle=True)
            le_individual_id = LabelEncoder()
            le_individual_id.classes_ = np.load(
                root / "individual_id.npy", allow_pickle=True
            )
            df["species_label"] = le_species.transform(df["species"])
            df["individual_id_label"] = le_individual_id.transform(df["individual_id"])
            if "valid" in pseudo_label_filename:
                df["fold"] = 0
            else:
                df["fold"] = -1
                df.index += 51033
            return df

        if phase != "test":
            df = pd.read_csv(str(root / "train.csv"))
            df_fb = pd.read_csv(str(root / "fullbody_train.csv"))
            df["bbox_fb"] = df_fb["bbox"].map(
                lambda x: [list(map(int, x[2:-2].split()))]
                if isinstance(x, str)
                else []
            )
            df["conf_fb"] = (
                df_fb["conf"]
                .map(lambda x: x if isinstance(x, str) else "[]")
                .map(eval)
                .map(lambda x: 0 if len(x) == 0 else x[0])
            )
            df_fb_charm = pd.read_csv(str(root / "fullbody_train_charm.csv"))
            df["bbox_fb_charm"] = df_fb_charm["bbox"].map(
                lambda x: [list(map(int, x[2:-2].split()))]
                if isinstance(x, str)
                else []
            )
            df["conf_fb_charm"] = (
                df_fb_charm["conf"]
                .map(lambda x: x if isinstance(x, str) else "[]")
                .map(eval)
                .map(lambda x: 0 if len(x) == 0 else x[0])
            )
            df_bbox_yolo = pd.read_csv(str(root / "train_bbox.csv"))
            df["bbox_yolo"] = df_bbox_yolo["bbox"].map(eval)
            df["conf_yolo"] = (
                df_bbox_yolo["conf"].map(eval).map(lambda x: 0 if len(x) == 0 else x[0])
            )
            df_bbox_detic = pd.read_csv(str(root / "train2.csv"))
            df["bbox_detic"] = df_bbox_detic["box"].map(
                lambda x: [list(map(int, x.split()))] if isinstance(x, str) else []
            )
            df["conf_detic"] = 1
            df_bbox_backfin = pd.read_csv(str(root / "train_backfin.csv"))
            df["bbox_backfin"] = df_bbox_backfin["bbox"].map(
                lambda x: [list(map(int, x[2:-2].split()))]
                if isinstance(x, str)
                else []
            )
            df["conf_backfin"] = (
                df_bbox_backfin["conf"]
                .map(lambda x: x if isinstance(x, str) else "[]")
                .map(eval)
                .map(lambda x: 0 if len(x) == 0 else x[0])
            )
            df_bbox_backfin_charm = pd.read_csv(str(root / "backfin_train_charm.csv"))
            df["bbox_backfin_charm"] = df_bbox_backfin_charm["bbox"].map(
                lambda x: [list(map(int, x[2:-2].split()))]
                if isinstance(x, str)
                else []
            )
            df["conf_backfin_charm"] = (
                df_bbox_backfin_charm["conf"]
                .map(lambda x: x if isinstance(x, str) else "[]")
                .map(eval)
                .map(lambda x: 0 if len(x) == 0 else x[0])
            )
        elif phase == "test":
            df = pd.read_csv(str(root / "sample_submission.csv"))
            df["species"] = 0
            df["species_label"] = 0
            df["individual_id_label"] = 0
            df_fb = pd.read_csv(str(root / "fullbody_test.csv"))
            df["bbox_fb"] = df_fb["bbox"].map(
                lambda x: [list(map(int, x[2:-2].split()))]
                if isinstance(x, str)
                else []
            )
            df["conf_fb"] = (
                df_fb["conf"]
                .map(lambda x: x if isinstance(x, str) else "[]")
                .map(eval)
                .map(lambda x: 0 if len(x) == 0 else x[0])
            )
            df_fb_charm = pd.read_csv(str(root / "fullbody_test_charm.csv"))
            df["bbox_fb_charm"] = df_fb_charm["bbox"].map(
                lambda x: [list(map(int, x[2:-2].split()))]
                if isinstance(x, str)
                else []
            )
            df["conf_fb_charm"] = (
                df_fb_charm["conf"]
                .map(lambda x: x if isinstance(x, str) else "[]")
                .map(eval)
                .map(lambda x: 0 if len(x) == 0 else x[0])
            )
            df_bbox_yolo = pd.read_csv(str(root / "test_bbox.csv"))
            df["bbox_yolo"] = df_bbox_yolo["bbox"].map(eval)
            df["conf_yolo"] = (
                df_bbox_yolo["conf"].map(eval).map(lambda x: 0 if len(x) == 0 else x[0])
            )
            df_bbox_detic = pd.read_csv(str(root / "test2.csv"))
            df["bbox_detic"] = df_bbox_detic["box"].map(
                lambda x: [list(map(int, x.split()))] if isinstance(x, str) else []
            )
            df["conf_detic"] = 1
            df_bbox_backfin = pd.read_csv(str(root / "test_backfin.csv"))
            df["bbox_backfin"] = df_bbox_backfin["bbox"].map(
                lambda x: [list(map(int, x[2:-2].split()))]
                if isinstance(x, str)
                else []
            )
            df["conf_backfin"] = (
                df_bbox_backfin["conf"]
                .map(lambda x: x if isinstance(x, str) else "[]")
                .map(eval)
                .map(lambda x: 0 if len(x) == 0 else x[0])
            )
            df_bbox_backfin_charm = pd.read_csv(str(root / "backfin_test_charm.csv"))
            df["bbox_backfin_charm"] = df_bbox_backfin_charm["bbox"].map(
                lambda x: [list(map(int, x[2:-2].split()))]
                if isinstance(x, str)
                else []
            )
            df["conf_backfin_charm"] = (
                df_bbox_backfin_charm["conf"]
                .map(lambda x: x if isinstance(x, str) else "[]")
                .map(eval)
                .map(lambda x: 0 if len(x) == 0 else x[0])
            )
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
        le_individual_id.classes_ = np.load(
            root / "individual_id.npy", allow_pickle=True
        )
        df["species_label"] = le_species.transform(df["species"])
        df["individual_id_label"] = le_individual_id.transform(df["individual_id"])

        if num_records:
            df = df[:num_records]

        return df

    def __init__(
        self, df: pd.DataFrame, phase="train", cfg=None, crop_aug=False
    ) -> None:
        self.df = df.copy()
        self.df["original_index"] = df.index

        self.df.reset_index(inplace=True)
        self.root = self.ROOT_PATH
        self.phase = phase
        self.crop = cfg.crop
        self.bbox = cfg.bbox
        self.bbox2 = cfg.bbox2
        self.crop_margin = cfg.crop_margin
        self.crop_aug = crop_aug
        self.p_fb = cfg.p_fb
        self.p_backfin = cfg.p_backfin
        self.p_detic = cfg.p_detic
        self.p_yolo = cfg.p_yolo
        self.p_fb_charm = cfg.p_fb_charm
        self.p_backfin_charm = cfg.p_backfin_charm
        self.p_none = cfg.p_none
        self.p_fb2 = cfg.p_fb2
        self.p_backfin2 = cfg.p_backfin2
        self.p_detic2 = cfg.p_detic2
        self.p_yolo2 = cfg.p_yolo2
        self.p_fb_charm2 = cfg.p_fb_charm2
        self.p_backfin_charm2 = cfg.p_backfin_charm2
        self.p_none2 = cfg.p_none2

        self.label_to_samples = {}
        for i in range(len(self.df)):
            label = self.df.at[i, "individual_id_label"]
            if label not in self.label_to_samples:
                self.label_to_samples[label] = []
            self.label_to_samples[label].append(i)

        self.label_names = list(self.label_to_samples.keys())

    def __len__(self) -> int:
        return len(self.df)

    def get_file_name(self, index: int, phase: str = "train") -> str:
        image_id = self.df.loc[index, "image"]
        if self.crop is not None:
            return f"cropped/cropped_{self.crop}_{phase}_images/{image_id}"

        return f"{phase}_images/{image_id}"

    def __getitem__(self, index: int):

        root = self.root

        if self.phase == "test" or self.df.at[index, "fold"] == -1:
            file_name = self.get_file_name(index, "test")
        else:
            file_name = self.get_file_name(index, self.phase)

        x = imageio.imread(root / file_name)
        x = np.asarray(x)

        if self.bbox is not None:
            crop_margin = self.crop_margin
            CONF = 0.01
            if self.crop_aug:
                bbox_type = ["fb", "backfin", "detic", "yolo", "fb_charm", "none"][
                    np.argmax(
                        np.random.multinomial(
                            1,
                            [
                                self.p_fb,
                                self.p_backfin,
                                self.p_detic,
                                self.p_yolo,
                                self.p_fb_charm,
                                self.p_none,
                            ],
                        )
                    )
                ]
            else:
                bbox_type = self.bbox
            if bbox_type == "none":
                bbox = []
                conf = 0
            else:
                bbox = self.df.at[index, f"bbox_{bbox_type}"]
                conf = self.df.at[index, f"conf_{bbox_type}"]
            if len(bbox) == 1 and conf >= CONF:
                xmin, ymin, xmax, ymax = bbox[0]
                dx = xmax - xmin
                dy = ymax - ymin
                xmin -= dx * crop_margin
                xmax += dx * crop_margin + 1
                ymin -= dy * crop_margin
                ymax += dy * crop_margin + 1

                size_x = x.shape[1]
                size_y = x.shape[0]
                xmin = int(max(0, xmin))
                xmax = int(min(xmax, size_x))
                ymin = int(max(0, ymin))
                ymax = int(min(ymax, size_y))
                image = x[ymin:ymax, xmin:xmax]
            else:
                image = x.copy()
        else:
            image = x.copy()

        if self.bbox2 is not None:
            crop_margin = self.crop_margin
            CONF = 0.01
            if self.crop_aug:
                bbox_type2 = ["fb", "backfin", "detic", "yolo", "fb_charm", "none"][
                    np.argmax(
                        np.random.multinomial(
                            1,
                            [
                                self.p_fb2,
                                self.p_backfin2,
                                self.p_detic2,
                                self.p_yolo2,
                                self.p_fb_charm2,
                                self.p_none2,
                            ],
                        )
                    )
                ]
            else:
                bbox_type2 = self.bbox2
            if bbox_type2 == "none":
                bbox = []
                conf = 0
            else:
                bbox = self.df.at[index, f"bbox_{bbox_type2}"]
                conf = self.df.at[index, f"conf_{bbox_type2}"]
            if len(bbox) == 1 and conf >= CONF:
                xmin, ymin, xmax, ymax = bbox[0]
                dx = xmax - xmin
                dy = ymax - ymin
                xmin -= dx * crop_margin
                xmax += dx * crop_margin + 1
                ymin -= dy * crop_margin
                ymax += dy * crop_margin + 1

                size_x = x.shape[1]
                size_y = x.shape[0]
                xmin = int(max(0, xmin))
                xmax = int(min(xmax, size_x))
                ymin = int(max(0, ymin))
                ymax = int(min(ymax, size_y))
                image2 = x[ymin:ymax, xmin:xmax]

            else:
                image2 = x.copy()

        res = {
            "original_index": self.df.at[index, "original_index"],
            "file_name": self.df.at[index, "image"],
            "label_species": self.df.at[index, "species_label"].astype(np.int64),
            "label": self.df.at[index, "individual_id_label"].astype(np.int64),
            "image": image,
        }

        if self.bbox2 is not None:
            res.update({"image2": image2})

        return res
