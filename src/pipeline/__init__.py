import os
import numpy as np
from PIL import Image
import cv2
from diffusers import StableDiffusionControlNetPipeline
from typing import Union, Literal

try:
    from src.utils import ImageProcessor

    base_path = '...'
except:
    from src.utils import ImageProcessor

    base_path = os.getcwd().split('src')[0]


def controlnet_landmark_pipeline(
        image_processor: ImageProcessor,
        pipeline: StableDiffusionControlNetPipeline,
        prompt: str = '',
        negative_prompt: str = '',
        image: Union[str, Image.Image, np.ndarray] = None,
        num_inference_steps: int = 50,
        controlnet_conditioning_scale: float = 1.0,
        guidance_scale: float = 5.0,
        input_type: Union[Literal['use_crop'], None] = None,
        output_type: Union[Literal['only_hull', 'only_crop_image'], None] = 'only_hull'):

    # crop face image & find landmark
    image_landmark, image_bbox = image_processor.get_face_crop_image(
        image=image, max_num_faces=1, ad=2, return_image=True)
    image_landmark, landmark_points = image_processor.get_facial_landmarks(
        image=image_landmark, max_num_faces=1, return_image=True)

    # run pipeline
    image_output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=256,
        width=256,
        image=image_landmark,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        cross_attention_kwargs={"scale": 1},
        guess_mode=False,
        guidance_scale=guidance_scale,
        return_dict=True
    ).images[0]

    # if type is 'only_crop_image', prepare face crop image
    if output_type == 'only_crop_image':
        return image_output, image_landmark

    # else, face crop image to original image
    image = image_processor.load_n_get_rgb(image, return_image=False)
    image_output = image_processor.load_n_get_rgb(image_output, return_image=False)
    image_output = cv2.resize(image_output, (int(image_bbox[2] - image_bbox[0]), int(image_bbox[3] - image_bbox[1])))

    # type_output 옵션이 only_hull일 경우 얼굴 영역만 추출하여 인페인팅, 아닐 경우 네모 모양으로 배경까지 변경됨
    if output_type == 'only_hull':
        image_crop = image[image_bbox[1]:image_bbox[3], image_bbox[0]:image_bbox[2]]
        hull = cv2.convexHull(np.array(landmark_points['outside']))
        mask = np.zeros(image_crop.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        masked_source = cv2.bitwise_and(image_output, image_output, mask=mask)
        masked_target = cv2.bitwise_and(image_crop, image_crop, mask=cv2.bitwise_not(mask))
        image_output = cv2.add(masked_source, masked_target)

    image[image_bbox[1]:image_bbox[3], image_bbox[0]:image_bbox[2]] = image_output
    image = Image.fromarray(image.astype('uint8'))

    return image, image_landmark
