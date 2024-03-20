import os
import datasets
import pandas as pd
from datasets import DownloadManager, DatasetInfo
from PIL import Image
import json
import io

# _URLS = {""}

class CelebHQLandmarkDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        # add number before name for sorting
        datasets.BuilderConfig(name="full"),
    ]

    def _info(self) -> DatasetInfo:
        features = {
            "crop_image": datasets.Image(),
            "landmark_image": datasets.Image(),
            "prompt_text": datasets.Value("string"),
            "mouth": datasets.Value("string"),
            "mouth_outside": datasets.Value("string"),
            "left_eye": datasets.Value("string"),
            "left_pupil": datasets.Value("string"),
            "right_eye": datasets.Value("string"),
            "right_pupil": datasets.Value("string"),
            "nose": datasets.Value("string"),
            "left_eyebrow": datasets.Value("string"),
            "left_eyebrow_up": datasets.Value("string"),
            "right_eyebrow": datasets.Value("string"),
            "right_eyebrow_up": datasets.Value("string"),
            "outside": datasets.Value("string"),
        }
        info = datasets.DatasetInfo(
            features=datasets.Features(features),
            supervised_keys=None,
            citation="",
        )
        return info

    def _split_generators(self, dl_manager: DownloadManager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": "/home/analysis02/tugboat-data/projectdata/celebahq/train",
                            }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": "/home/analysis02/tugboat-data/projectdata/celebahq/valid",
                            }
            ),
        ]

    def _generate_examples(self, path):
        with open(path + '/metadata.json', 'r', encoding='utf-8') as file:
            metadata = json.load(file)
        metadata = pd.DataFrame(metadata)
        print(metadata.head(5))

        for index, item in metadata.iterrows():

            with Image.open(path + '/' + item["crop"]) as crop_img, Image.open(path + '/' + item["landmark"]) as landmark_img:

                crop_img_byte = io.BytesIO()
                landmark_img_byte = io.BytesIO()
                crop_img.save(crop_img_byte, format=crop_img.format)
                landmark_img.save(landmark_img_byte, format=landmark_img.format)

                yield index, {
                    "crop_image": crop_img_byte.getvalue(),
                    "landmark_image": landmark_img_byte.getvalue(),
                    "prompt_text": item["text"],
                    "mouth": json.dumps(item["landmark_point"]["mouth"]),
                    "mouth_outside": json.dumps(item["landmark_point"]["mouth_outside"]),
                    "left_eye": json.dumps(item["landmark_point"]["left_eye"]),
                    "left_pupil": json.dumps(item["landmark_point"]["left_pupil"]),
                    "right_eye": json.dumps(item["landmark_point"]["right_eye"]),
                    "right_pupil": json.dumps(item["landmark_point"]["right_pupil"]),
                    "nose": json.dumps(item["landmark_point"]["nose"]),
                    "left_eyebrow": json.dumps(item["landmark_point"]["left_eyebrow"]),
                    "left_eyebrow_up": json.dumps(item["landmark_point"]["left_eyebrow_up"]),
                    "right_eyebrow": json.dumps(item["landmark_point"]["right_eyebrow"]),
                    "right_eyebrow_up": json.dumps(item["landmark_point"]["right_eyebrow_up"]),
                    "outside": json.dumps(item["landmark_point"]["outside"]),
                }