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
            "default": datasets.Image(),
            "landmark": datasets.Image(),
            "inpaint": datasets.Image(),
            "de_identification": datasets.Image(),
            "text": datasets.Value("string"),
            "landmarks": datasets.Value("string"),
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

            with (
                Image.open(path + '/' + item["default"]) as default_img,
                Image.open(path + '/' + item["landmark"]) as landmark_img,
                Image.open(path + '/' + item["inpaint"]) as inpaint_img,
                Image.open(path + '/' + item["de_identification"]) as de_identification_img
            ):

                default_img_byte = io.BytesIO()
                landmark_img_byte = io.BytesIO()
                inpaint_img_byte = io.BytesIO()
                de_identification_img_byte = io.BytesIO()

                default_img.save(default_img_byte, format=default_img.format)
                landmark_img.save(landmark_img_byte, format=landmark_img.format)
                inpaint_img.save(inpaint_img_byte, format=inpaint_img.format)
                de_identification_img.save(de_identification_img_byte, format=de_identification_img.format)

                yield index, {
                    "default": default_img_byte.getvalue(),
                    "landmark": landmark_img_byte.getvalue(),
                    "inpaint": inpaint_img_byte.getvalue(),
                    "de_identification": de_identification_img_byte.getvalue(),
                    "text": item["text"],
                    "landmarks": item["landmark_point"], # json.dumps(item["landmark_point"]),
                }