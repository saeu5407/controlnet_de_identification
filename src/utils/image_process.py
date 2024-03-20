from PIL import Image
import numpy as np
import cv2
from transformers import pipeline
from typing import Optional, Union, List, Any, Callable, Dict
import torch
import json

class ImageProcesserV1():
    def __init__(self,
                 device: Union[torch.device, None] = None,
                 height: Optional[int] = 512,
                 width: Optional[int] = 512):
        super(ImageProcesserV1, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.openpose = None
        self.height = height
        self.width = width
        print("set image processer")

    def get_rgb_from_rgba(self,
                          image: Union[np.ndarray, Image.Image]):
        image_array = np.array(image)
        rgb_image_array = image_array[:, :, :3]
        # TODO: 반투명도 조절하는 걸로 함수 변경해야 함
        return Image.fromarray(rgb_image_array)

    def open(self,
             image_path: Union[str, Image.Image]):
        image = Image.open(image_path) if type(image_path) == str else image_path
        if np.array(image).shape[2] == 4:
            image = self.get_rgb_from_rgba(image)
        return image

    def preprocess(self,
                   image_path: Union[str, Image.Image, np.ndarray],
                   dtype: Union[torch.dtype, None],
                   do_classifier_free_guidance: Optional[bool] = False,
                   height: Optional[int] = None,
                   width: Optional[int] = None,
                   ):
        if height is None:
            height = self.height
        if width is None:
            width = self.width

        if type(image_path) != np.ndarray:
            image = self.open(image_path)
            image = np.array(image)
        else:
            image = image_path

        print(image.shape)
        print(width, height)
        image = cv2.resize(image, dsize=(width, height))

        image = torch.from_numpy(image).unsqueeze(0)
        image = image.type(dtype)
        image /= 255.0
        image = image.permute(0, 3, 1, 2)
        image = image.to(self.device)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def postprocess(self,
                    output: Union[np.ndarray, torch.Tensor]):
        if type(output) == np.ndarray:
            output = torch.tensor(output)
        output = (output / 2 + 0.5).clamp(0, 1)
        output = output.cpu().permute(0, 2, 3, 1).float().numpy()
        output = (output * 255).round().astype("uint8")
        output = Image.fromarray(output[0])
        return output

    def get_canny_edges(self,
                        image: Union[np.ndarray, Image.Image],
                        low_threshold: Optional[int] = 100,
                        high_threshold: Optional[int] = 200,
                        return_image: Optional[bool] = False):
        if type(image) == Image.Image:
            image = np.array(image)
        edges = cv2.Canny(image, low_threshold, high_threshold)
        edges = edges[:, :, None]
        edges = np.concatenate([edges, edges, edges], axis=2)
        if return_image:
            edges = Image.fromarray(edges)
        return edges

    # TODO: get_depth_map 인풋 타입 설정 및 이에 맞춰 코드 수정 필요
    def get_depth_map(self, image, depth_estimator=pipeline("depth-estimation")):
        depth = depth_estimator(image)["depth"]
        depth = np.array(depth)
        depth = depth[:, :, None]
        depth_map = np.concatenate([depth, depth, depth], axis=2)
        return Image.fromarray(depth_map)

    def get_openpose(self,
                     image: Union[np.ndarray, Image.Image],
                     ):
        if self.openpose is None:
            from ..annotator.openpose import OpenposeDetector
            self.openpose = OpenposeDetector()

        if type(image) is Image.Image:
            image = self.open(image)
            image = np.array(image)

        image = HWC3(image)
        image, _ = self.openpose(resize_image(image, image.shape[0]))
        image = HWC3(image)
        return image

class ImageProcessor():
    def __init__(self,
                 device: Union[torch.device, None] = None,
                 resize: Union[tuple, list] = (512, 512),
                 facial_landmarks: Optional[bool] = False,
                 face_detection: Optional[bool] = False,
                 ):
        super(ImageProcessor, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resize = resize
        self.facial_landmarks = facial_landmarks
        self.face_detection = face_detection

        h, w = resize
        h_resize = int(h // 64 * 64)
        w_resize = int(w // 64 * 64)
        if (h_resize != h) or (w_resize != w):
            print(f"Warning: Your resize input is : {(h, w)} but change to {(h_resize, w_resize)}")
            self.resize = (h_resize, w_resize)

        if self.facial_landmarks:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)

        if self.face_detection:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detect = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def load_n_get_rgb(self,
                       image: Union[np.ndarray, Image.Image, str] = None,
                       bgr2rgb: Optional[bool] = False,
                       return_image: Optional[bool] = False,
                       ):
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except FileNotFoundError:
                raise FileNotFoundError(f"Error: Not found image, check : {image}")

        if isinstance(image, Image.Image):
            image = np.array(image)

        if bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W, C = image.shape
        if C == 1:
            image = np.concatenate([image, image, image], axis=2)
        if C == 4:
            color = image[:, :, 0:3].astype(np.float32)
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            image = color * alpha + 255.0 * (1.0 - alpha)
            image = image.clip(0, 255).astype(np.uint8)

        if return_image:
            image = Image.fromarray(image)

        return image

    def preprocess(self,
                   image: Optional[np.ndarray],
                   dtype: Union[torch.dtype, None] = torch.float32,
                   do_classifier_free_guidance: Optional[bool] = False,
                   ):
        """
        처리된 np.array 형태의 이미지를 입력받아서 Stable Diffusion을 위한 전처리 수행
        :param image: np.array
        :param dtype: 목표로 할 데이터 타입
        :param do_classifier_free_guidance: Classfier Free Guidance 여부
        :return: 전처리 된 이미지
        """
        if self.resize is not None:
            image = cv2.resize(image, dsize=self.resize)

        image = torch.from_numpy(image).unsqueeze(0)
        image = image.type(dtype)
        image /= 255.0
        image = image.permute(0, 3, 1, 2)
        image = image.to(self.device)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def postprocess(self,
                    output: Union[np.ndarray, torch.Tensor],
                    return_image: Optional[bool] = True
                    ):
        """
        모델에서 출력된 이미지 값에 대해 후처리 수행
        :param output: 모델에서 출력된 이미지 값
        :param return_image: True 일 시 배치의 맨 첫 이미지에 대해 Image.Image로 변환
        :return:
        """
        if isinstance(output, np.ndarray):
            output = torch.tensor(output)
        output = (output / 2 + 0.5).clamp(0, 1)
        output = output.cpu().permute(0, 2, 3, 1).float().numpy()
        output = (output * 255).round().astype("uint8")

        if return_image:
            output = Image.fromarray(output[0])

        return output

    def get_facial_landmarks(self,
                             image: Union[np.ndarray, str, Image.Image],
                             use_segmentation_map: Optional[bool] = True,
                             use_facial_landmark: Optional[bool] = True,
                             return_image: Optional[bool] = True,
                             max_num_faces: Union[int, None] = None):

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.load_n_get_rgb(image=image)

        # 얼굴 랜드마크 검출
        results = self.face_mesh.process(image)

        # 랜드마크 포인트
        """https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png"""
        outside = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                   176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        nose = [6, 197, 195, 5, 4]
        mouth = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        mouth_outside = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

        left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        left_pupil = [469, 470, 471, 472]
        left_eyebrow = [46, 53, 52, 65, 55]
        left_eyebrow_up = [70, 63, 105, 66, 107]

        right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        right_pupil = [474, 475, 476, 477]
        right_eyebrow = [285, 295, 282, 283, 276]
        right_eyebrow_up = [336, 296, 334, 293, 300]

        landmark_list = []

        # 검출된 얼굴에 대해 마스크 생성
        if results.multi_face_landmarks:
            len_result = len(results.multi_face_landmarks)

            if max_num_faces is not None:
                len_result = min(int(max_num_faces), len_result)

            height, width, _ = image.shape
            mask = np.zeros((height, width), dtype=np.uint8)

            # 루프로 해 두었지만 랜드마크 처리도 해야되니 그냥 하나라고 생각하자
            for face_landmarks in results.multi_face_landmarks[:len_result]:

                landmark_dict = {}
                points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks.landmark]

                if use_segmentation_map:
                    i_poly = np.array([points[i] for i in outside], dtype=np.int32)
                    hull = cv2.convexHull(i_poly)
                    cv2.fillConvexPoly(mask, hull, 255)
                    image[mask == 255] = (0, 0, 0)
                    landmark_dict['outside'] = i_poly.tolist()

                if use_facial_landmark:
                    # mouth
                    mouth_poly = np.array([points[i] for i in mouth], dtype=np.int32)
                    mouth_outside_poly = np.array([points[i] for i in mouth_outside], dtype=np.int32)
                    cv2.polylines(image, [mouth_poly], True, (255, 255, 0), 1, cv2.LINE_AA)
                    cv2.polylines(image, [mouth_outside_poly], True, (255, 255, 0), 1, cv2.LINE_AA)
                    landmark_dict['mouth'] = mouth_poly.tolist()
                    landmark_dict['mouth_outside'] = mouth_outside_poly.tolist()

                    # left
                    # TODO: pupil이 지금은 직사각형 poly draw라 나중에 개선 필요
                    left_eye_poly = np.array([points[i] for i in left_eye], dtype=np.int32)
                    left_pupil_poly = np.array([points[i] for i in left_pupil], dtype=np.int32)
                    cv2.polylines(image, [left_eye_poly], True, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.polylines(image, [left_pupil_poly], True, (0, 255, 0), 1, cv2.LINE_AA)
                    landmark_dict['left_eye'] = left_eye_poly.tolist()
                    landmark_dict['left_pupil'] = left_pupil_poly.tolist()

                    # right
                    right_eye_poly = np.array([points[i] for i in right_eye], dtype=np.int32)
                    right_pupil_poly = np.array([points[i] for i in right_pupil], dtype=np.int32)
                    cv2.polylines(image, [right_eye_poly], True, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.polylines(image, [right_pupil_poly], True, (0, 0, 255), 1, cv2.LINE_AA)
                    landmark_dict['right_eye'] = right_eye_poly.tolist()
                    landmark_dict['right_pupil'] = right_pupil_poly.tolist()

                    # nose
                    nose_poly = np.array([points[i] for i in nose], dtype=np.int32)
                    for i in range(len(nose_poly[:-1])):
                        cv2.line(image, nose_poly[i], nose_poly[i+1], (255, 0, 0), 1, cv2.LINE_AA)
                    landmark_dict['nose'] = nose_poly.tolist()

                    # left eyebrow
                    left_eyebrow_poly = np.array([points[i] for i in left_eyebrow], dtype=np.int32)
                    left_eyebrow_up_poly = np.array([points[i] for i in left_eyebrow_up], dtype=np.int32)
                    for i in range(len(left_eyebrow_poly[:-1])):
                        cv2.line(image, left_eyebrow_poly[i], left_eyebrow_poly[i+1], (0, 255, 255), 1, cv2.LINE_AA)
                    for i in range(len(left_eyebrow_up_poly[:-1])):
                        cv2.line(image, left_eyebrow_up_poly[i], left_eyebrow_up_poly[i+1], (0, 255, 255), 1, cv2.LINE_AA)
                    landmark_dict['left_eyebrow'] = left_eyebrow_poly.tolist()
                    landmark_dict['left_eyebrow_up'] = left_eyebrow_up_poly.tolist()

                    # right eyebrow
                    right_eyebrow_poly = np.array([points[i] for i in right_eyebrow], dtype=np.int32)
                    right_eyebrow_up_poly = np.array([points[i] for i in right_eyebrow_up], dtype=np.int32)
                    for i in range(len(right_eyebrow_poly[:-1])):
                        cv2.line(image, right_eyebrow_poly[i], right_eyebrow_poly[i+1], (255, 0, 255), 1, cv2.LINE_AA)
                    for i in range(len(right_eyebrow_up_poly[:-1])):
                        cv2.line(image, right_eyebrow_up_poly[i], right_eyebrow_up_poly[i+1], (255, 0, 255), 1, cv2.LINE_AA)
                    landmark_dict['right_eyebrow'] = right_eyebrow_poly.tolist()
                    landmark_dict['right_eyebrow_up'] = right_eyebrow_up_poly.tolist()

                    landmark_list.append(landmark_dict)
            if return_image:
                image = Image.fromarray(image)
                landmark_list = landmark_list[0]
        else:
                image = None
                landmark_list = []

        return image, landmark_list

    def get_face_crop_image(self,
                            image: Union[np.ndarray, str, Image.Image],
                            ad: Optional[float] = 1,
                            return_image: Optional[bool] = True,
                            max_num_faces: Union[int, None] = None):

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.load_n_get_rgb(image=image)

        images = image
        face_list = []
        face_bbox_list = []

        results = self.face_detect.process(image)
        width, height = image.shape[1:3] if len(image.shape) == 4 else image.shape[0:2]

        if results.detections:
            len_result = len(results.detections)

            if max_num_faces is not None:
                len_result = min(int(max_num_faces), len_result)

            for detection in results.detections[:len_result]:
                # 기본 박스 좌표 계산
                bboxC = detection.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * width)
                ymin = int(bboxC.ymin * height)
                bbox_width = int(bboxC.width * width)
                bbox_height = int(bboxC.height * height)

                # 스케일링 변수를 적용하여 박스 크기 조정
                xmin_scaled = max(0, int(xmin - bbox_width * (ad - 1) / 2))
                ymin_scaled = max(0, int(ymin - bbox_height * (ad - 1) / 2))
                xmax_scaled = min(width, int(xmin + bbox_width + bbox_width * (ad - 1) / 2))
                ymax_scaled = min(height, int(ymin + bbox_height + bbox_height * (ad - 1) / 2))

                # 수정된 박스로 얼굴 부분 추출
                face_list.append(image[ymin_scaled:ymax_scaled, xmin_scaled:xmax_scaled])
                face_bbox_list.append([xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled])

            images = face_list[0] if len(face_list) == 1 else np.stack(face_list)
            bboxs = face_bbox_list[0] if len(face_list) == 1 else np.stack(face_bbox_list)

            if return_image and (len(face_list) <= 1):
                images = Image.fromarray(images)
            else:
                print("Warning: Image have more than one faces")
        else:
            images = None
            bboxs = []

        return images, bboxs

if __name__ == '__main__':

    image_processor = ImageProcessor(facial_landmarks=True, face_detection=True)
    image_path = '../dataset/sample2.png'

    facial_landmark_image, landmark_points = image_processor.get_facial_landmarks(image=image_path, max_num_faces=1, return_image=True)
    facial_landmark_image.save('../../output/sample_facial_landmark.png')
    with open('../../output/landmark.json', 'w') as f:
        json.dump(landmark_points, f)

    face_detection_image, bboxs = image_processor.get_face_crop_image(image=image_path, max_num_faces=1, ad=2, return_image=True)
    print(bboxs)
    face_detection_image.save('../../output/sample_face_detection.png')