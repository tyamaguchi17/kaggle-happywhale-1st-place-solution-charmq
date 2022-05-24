from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from src.utils.tta import TestTimeAugmentor


class Forwarder(nn.Module):
    def __init__(self, cfg: DictConfig, model: nn.Module) -> None:
        super().__init__()
        self.model = model

        self.tta = TestTimeAugmentor(
            flip_h=cfg.tta.flip_h, rot90=cfg.tta.rot90, rot180=cfg.tta.rot180
        )
        self.head = cfg.head.type
        self.head_species = cfg.head.head_species
        self.backbone2 = cfg.backbone2
        self.input_species = cfg.species_embedding_size > 0

    def loss(
        self,
        logits_margin,
        labels,
        mean=True,
        logits=None,
        embed_features=None,
        logits_species=None,
        species_label=None,
        epoch=None,
    ) -> Tensor:
        loss = F.cross_entropy(logits_margin, labels, reduction="none")  # (B, C)
        if logits_species is not None:
            loss += F.cross_entropy(logits_species, species_label, reduction="none")
        if mean:
            return torch.mean(loss)
        else:
            return loss

    def forward(
        self, batch: Dict[str, Tensor], phase: str, epoch=None
    ) -> Tuple[Tensor, Tensor]:

        # inputs: Input tensor.
        inputs = batch["image"]
        if self.backbone2:
            c = inputs.shape[1]
            inputs, inputs2 = inputs[:, : c // 2], inputs[:, c // 2 :]

        # labels: Target labels of shape (B, C) where C is the number of classes.
        labels = batch["label"]
        species_label = batch["label_species"]

        if phase == "train":
            with torch.set_grad_enabled(True):
                embed_features = self.model.forward_features(inputs)
                if self.backbone2:
                    embed_features2 = self.model.forward_features2(inputs2)
                    embed_features = torch.cat([embed_features, embed_features2], dim=1)
                if self.input_species:
                    embed_species = self.model.species_embedding(species_label)
                    embed_features = torch.cat([embed_features, embed_species], dim=1)
                logits_margin, logits = self.model.head(embed_features, labels)
                if self.head_species:
                    logits_species_margin, logits_species = self.model.head_species(
                        embed_features, species_label
                    )
                else:
                    logits_species = None
                    logits_species_margin = None
                embed_features1 = embed_features
                embed_features2 = embed_features
            loss = self.loss(
                logits_margin=logits_margin,
                labels=labels,
                logits=logits,
                embed_features=embed_features,
                logits_species=logits_species_margin,
                species_label=species_label,
                epoch=epoch,
            )
        else:
            if phase == "test":
                inputs_tta = self.tta.get_inputs(inputs)
                (embed_features_tta,) = self.tta.run(
                    self.model.forward_features, inputs_tta
                )
                embed_features1 = embed_features_tta[0]
                embed_features2 = embed_features_tta[1]
                embed_features = torch.mean(embed_features_tta, dim=0)
                if self.backbone2:
                    inputs_tta2 = self.tta.get_inputs(inputs2)
                    (embed_features_tta2,) = self.tta.run(
                        self.model.forward_features2, inputs_tta2
                    )
                    embed_features3 = embed_features_tta2[0]
                    embed_features4 = embed_features_tta2[1]
                    embed_features1 = torch.cat(
                        [embed_features1, embed_features3], dim=1
                    )
                    embed_features2 = torch.cat(
                        [embed_features2, embed_features4], dim=1
                    )
                    embed_features = (embed_features1 + embed_features2) / 2
                if self.input_species:
                    embed_species = self.model.species_embedding(species_label)
                    embed_features = torch.cat([embed_features, embed_species], dim=1)
            else:
                embed_features = self.model.forward_features(inputs)
                if self.backbone2:
                    embed_features2 = self.model.forward_features2(inputs2)
                    embed_features = torch.cat([embed_features, embed_features2], dim=1)
                if self.input_species:
                    embed_species = self.model.species_embedding(species_label)
                    embed_features = torch.cat([embed_features, embed_species], dim=1)
                embed_features1 = embed_features
                embed_features2 = embed_features
            logits_margin, logits = self.model.head(embed_features, labels)
            if self.head_species:
                logits_species_margin, logits_species = self.model.head_species(
                    embed_features, species_label
                )
            else:
                logits_species = None
                logits_species_margin = None
            loss = self.loss(
                logits_margin=logits_margin,
                labels=labels,
                logits=logits,
                embed_features=embed_features,
                logits_species=logits_species_margin,
                species_label=species_label,
                epoch=epoch,
            )

        return logits, loss, embed_features1, logits_species, embed_features2
