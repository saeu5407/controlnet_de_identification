import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Optional, Union
import fire

try:
    from ..utils import ImageProcessor
except:
    from src.utils import ImageProcessor

def make_crop_landmark_dataset(
        dataset_path: Optional[str] = '/home/analysis02/tugboat-data/projectdata/celebahq256_imgs/train',
        output_path: Optional[str] = '/home/analysis02/tugboat-data/projectdata/celebahq/train',
        resize: Union[tuple, list] = (256, 256)
):

    dataset_path = Path(dataset_path)
    dataset_list = list(dataset_path.glob('*.jpg')) + \
                   list(dataset_path.glob('*.jpeg')) + \
                   list(dataset_path.glob('*.png'))

    os.makedirs(output_path + '/crop', exist_ok=True)
    os.makedirs(output_path + '/landmark', exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_processor = ImageProcessor(facial_landmarks=True, face_detection=True)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()

    raw_data = []
    for i, idx in tqdm(enumerate(dataset_list), total=len(dataset_list)):

        # set path
        # {i:06d}
        crop_path = 'crop' + f'/{i}.png'
        landmark_path = 'landmark' + f'/{i}.png'

        # crop image
        image, _ = image_processor.get_face_crop_image(image=str(idx), max_num_faces=1, ad=2, return_image=True)

        if image is not None:
            crop_image = cv2.resize(np.array(image), resize)

            # get caption using blip
            '''
            # blip에 prompt를 넣어도 좋은 결과가 나오지 않음.
            prompt_text = 'provide detailed descriptions of expression, ethnicity, age, gender'
            prompt_text = 'what type of clothing is the person wearing'
            inputs = processor(text=[prompt_text], images=np.array(image), return_tensors="pt").to(device)
            '''
            # 이 부분을 chatgpt를 사용해서 생성하고 결과 텍스트를 저장해서 나중에 blip을 파인튜닝하는 방안 고려
            inputs = processor(images=np.array(image), return_tensors="pt").to(device)
            outputs = model.generate(**inputs)
            text = processor.decode(outputs[0], skip_special_tokens=True)

            # visualize facial landmark
            image, landmark = image_processor.get_facial_landmarks(image=image, max_num_faces=1, return_image=True)

            if image is not None:
                landmark_image = cv2.resize(np.array(image), (256, 256))

                # append
                Image.fromarray(crop_image).save(output_path + '/' + crop_path)
                Image.fromarray(landmark_image).save(output_path + '/' + landmark_path)
                raw_data.append([crop_path, landmark_path, text, landmark])

    raw_data = pd.DataFrame(raw_data, columns=['crop', 'landmark', 'text', 'landmark_point'])
    raw_data.to_json(output_path + '/metadata.json')

if __name__ == '__main__':

    fire.Fire(make_crop_landmark_dataset)
