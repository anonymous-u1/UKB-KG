# entity_type_tag/infer.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict

LABEL_Y_MAP = {
    "ACTI": 0,
    "ANAT": 1,
    "BASE": 2,
    "CHEM": 3,
    "DISO": 4,
    "GENE": 5,
    "MEAS": 6,
    "MISC": 7,
    "PHYS": 8,
    "PROC": 9,
}
ID2LABEL = {v: k for k, v in LABEL_Y_MAP.items()}


class EntityTypeClassifier:
    def __init__(
        self,
        model_dir: str = "entity_type_tag/save/save_35w_256",
        device: str = "cuda",
        max_length: int = 64,
        batch_size: int = 64,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, entities: List[str]) -> List[str]:
        """
        输入 entity 字符串列表，返回预测的大类标签列表
        """
        results = []

        for i in range(0, len(entities), self.batch_size):
            batch = entities[i: i + self.batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            results.extend([ID2LABEL[p] for p in preds])

        return results

    def predict_dict(self, entities: List[str]) -> Dict[str, str]:
        """
        返回 {entity_name: label}
        """
        labels = self.predict(entities)
        return dict(zip(entities, labels))

# 使用示例
# from entity_type_tag.entity_type import EntityTypeClassifier
#
# clf = EntityTypeClassifier(model_dir="entity_type_tag/save/save_10w_256")
# label = clf.predict(["Microalbuminuria", "Microalbumin"])
# print(label)