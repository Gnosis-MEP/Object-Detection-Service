import os
import zipfile

import six.moves.urllib as urllib

import numpy as np
import tensorflow as tf
import cv2

from tf_od_models.object_detection.coco_based_od import COCOBasedModel


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def download_model(model_name=None, model_path=None):
    model_file_name = model_name + '.zip'
    model_file_path = os.path.join(CURRENT_DIR, 'models', model_file_name)

    if model_path is None:
        model_path = os.path.join(CURRENT_DIR, 'models', f'{model_name}.meta')

    if not os.path.isfile(model_path):
        url = f"https://www.dropbox.com/s/qhvrkr25abnplhu/{model_file_name}?dl=1"
        print(f'Model not found. Downloading it now:{url, model_file_path}')
        u = urllib.request.urlopen(url)
        data = u.read()
        u.close()
        with open(model_file_path, "wb") as f:
            f.write(data)

        extract_location = os.path.join(CURRENT_DIR, 'models')
        zip_ref = zipfile.ZipFile(model_file_path)  # create zipfile object
        zip_ref.extractall(extract_location)  # extract file to dir
        zip_ref.close()  # close file
        os.remove(model_file_path)

    else:
        print('Model found. Proceed.')


class COCOHatBasedModel(COCOBasedModel):
    # def __init__(self, base_configs, lazy_setup=False):
    #     super(COCOHatBasedModel, self).__init__()
    #     self.base_configs = base_configs
    #     if not lazy_setup:
    #         self.setup()

    def setup(self):
        model_name = self.base_configs['model_name']
        # width = self.base_configs['width']
        # height = self.base_configs['height']
        # # detection_threshold = 0.01
        # detection_threshold = self.base_configs['detection_threshold']
        # allow_memory_growth = self.base_configs['allow_memory_growth']
        # tf_gpu_fraction = self.base_configs['tf_gpu_fraction']
        download_model(model_name=model_name)
        # self.model_configs = get_session_and_model_ready(
        #     model_name=model_name, width=width, height=height,
        #     detection_threshold=detection_threshold,
        #     allow_memory_growth=allow_memory_growth, tf_gpu_fraction=tf_gpu_fraction)
        # if self.base_configs['hot_start'] is True:
        #     print('Running hot start...')
        #     self._hot_start(width, height)
        #     print('Finished hot start...')

    # def predict(self, input_image):
    #     origin_height, origin_width = input_image.shape[:-1]
    #     category_index = self.model_configs['category_index']
    #     detection_threshold = self.model_configs['detection_threshold']
    #     width = self.base_configs['width']
    #     height = self.base_configs['height']
    #     resized_image = resize_image(input_image, height, width)
    #     preprocessed_image = preprocessing(resized_image)
    #     (boxes, scores, classes, num) = predict(self.model_configs, preprocessed_image)
    #     detections = post_processing(
    #         boxes, scores, classes, detection_threshold, category_index,
    #         origin_height, origin_width
    #     )
    #     return detections

    # def add_bbboxes_to_image(self, input_image, detections):
    #     im_height, im_width = input_image.shape[:-1]
    #     output_image = input_image
    #     for detection in detections['data']:
    #         label = detection['label']
    #         confidence = detection['confidence']
    #         bbox = detection['bounding_box']
    #         color = (254.0, 254.0, 254)
    #         output_image = cv2.rectangle(
    #             output_image,
    #             (bbox[0], bbox[1]), (bbox[2], bbox[3]),
    #             color
    #         )
    #         label_conf = f'{label}: {confidence}'
    #         cv2.putText(output_image, label_conf, (bbox[0] - 10, bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #     return output_image

if __name__ == '__main__':
    model = COCOHatBasedModel(
        base_configs={
            'model_name': 'hat_not_hat'
        }
    )