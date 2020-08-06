#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tarfile

import six.moves.urllib as urllib

import numpy as np
import tensorflow as tf
import cv2

from object_detection.utils import label_map_util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def download_model(model_name=None, model_path=None):
    # print(f'CURRENT_DIR: {CURRENT_DIR}')
    if model_name is None:
        model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    if model_path is None:
        model_path = os.path.join(CURRENT_DIR, 'models', model_name, 'frozen_inference_graph.pb')

    model_file_name = model_name + '.tar.gz'
    model_file_path = os.path.join(CURRENT_DIR, 'models', model_file_name)

    download_base = 'http://download.tensorflow.org/models/object_detection/'
    if not os.path.isfile(model_path):
        print(f'Model not found. Downloading it now:{download_base + model_file_name, model_file_path}')
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file_name, model_file_path)
        tar_file = tarfile.open(model_file_path)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.path.join(CURRENT_DIR, 'models'))
        os.remove(model_file_path)
    else:
        print('Model found. Proceed.')


def load_frozenmodel(model_name=None, model_path=None, label_path=None, num_classes=90):
    if model_name is None:
        model_name = 'ssd_mobilenet_v11_coco'
    if model_path is None:
        model_path = os.path.join(CURRENT_DIR, 'models', model_name, 'frozen_inference_graph.pb')
    if label_path is None:
        label_path = os.path.join(CURRENT_DIR, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    # Loading label map
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return detection_graph, category_index


def get_session_and_model_ready(model_name, width=300, height=300,
                                detection_threshold=0.5,
                                allow_memory_growth=True, tf_gpu_fraction=1.0):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=tf_gpu_fraction, allow_growth=allow_memory_growth)
    detection_graph, category_index = load_frozenmodel(model_name=model_name)
    sess = tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options))
    sess.__enter__()
    tf.global_variables_initializer().run(session=sess)
    model_configs = {
        'session': sess,
        'image_tensor': detection_graph.get_tensor_by_name('image_tensor:0'),
        # Each box represents a part of the image where a particular object was detected.
        'detection_boxes': detection_graph.get_tensor_by_name('detection_boxes:0'),
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        'detection_scores': detection_graph.get_tensor_by_name('detection_scores:0'),
        'detection_classes': detection_graph.get_tensor_by_name('detection_classes:0'),
        'num_detections': detection_graph.get_tensor_by_name('num_detections:0'),
        'category_index': category_index,
        'detection_threshold': detection_threshold
    }
    return model_configs


def resize_image(input_image, input_height, input_width):
    resized_image = cv2.resize(
        input_image, (input_height, input_width), interpolation=cv2.INTER_CUBIC)
    return resized_image


def preprocessing(input_image):
    # Add the dimension relative to the batch size needed for the input placeholder "x"
    image_array = np.expand_dims(input_image, axis=0)  # Add batch dimension

    return image_array


def predict(model_configs, preprocessed_image):
    detection_boxes = model_configs.get('detection_boxes')
    detection_scores = model_configs.get('detection_scores')
    detection_classes = model_configs.get('detection_classes')
    num_detections = model_configs.get('num_detections')
    image_tensor = model_configs.get('image_tensor')

    sess = model_configs.get('session')
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: preprocessed_image}
    )
    return (boxes, scores, classes, num)


def post_processing(boxes, scores, classes, detection_threshold, category_index, origin_height, origin_width):
    output = []
    for bbox, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
        if score > detection_threshold:
            label = category_index[_class]['name']
            norm_bbox = normalized_bbox(bbox, origin_height, origin_width)
            obj = {
                'label': label,
                'bounding_box': norm_bbox,
                'confidence': float(score)
            }
            output.append(obj)
    return {'data': output}


def normalized_bbox(bbox, origin_height, origin_width):
    # rescale the coordinates to the original image
    ymin, xmin, ymax, xmax = bbox
    (left, right, top, bottom) = (xmin * origin_width, xmax * origin_width,
                                  ymin * origin_height, ymax * origin_height)
    return (int(left), int(top), int(right), int(bottom))


class COCOBasedModel(object):
    def __init__(self, base_configs, lazy_setup=False):
        super(COCOBasedModel, self).__init__()
        self.base_configs = base_configs
        if not lazy_setup:
            self.setup()

    def _hot_start(self, width, height, rgb_color=(0, 0, 0)):
        # Create black blank image
        image = np.zeros((height, width, 3), np.uint8)
        # Since OpenCV uses BGR, convert the color first
        color = tuple(reversed(rgb_color))
        # Fill image with color
        image[:] = color
        return self.predict(image)

    def setup(self):
        model_name = self.base_configs['model_name']
        width = self.base_configs['width']
        height = self.base_configs['height']
        # detection_threshold = 0.01
        detection_threshold = self.base_configs['detection_threshold']
        allow_memory_growth = self.base_configs['allow_memory_growth']
        tf_gpu_fraction = self.base_configs['tf_gpu_fraction']
        download_model(model_name=model_name)
        self.model_configs = get_session_and_model_ready(
            model_name=model_name, width=width, height=height,
            detection_threshold=detection_threshold,
            allow_memory_growth=allow_memory_growth, tf_gpu_fraction=tf_gpu_fraction)
        if self.base_configs['hot_start'] is True:
            print('Running hot start...')
            self._hot_start(width, height)
            print('Finished hot start...')

    def predict(self, input_image):
        origin_height, origin_width = input_image.shape[:-1]
        category_index = self.model_configs['category_index']
        detection_threshold = self.model_configs['detection_threshold']
        width = self.base_configs['width']
        height = self.base_configs['height']
        resized_image = resize_image(input_image, height, width)
        preprocessed_image = preprocessing(resized_image)
        (boxes, scores, classes, num) = predict(self.model_configs, preprocessed_image)
        detections = post_processing(
            boxes, scores, classes, detection_threshold, category_index,
            origin_height, origin_width
        )
        return detections

    def stop(self):
        self.model_configs['session'].close()

    def add_bbboxes_to_image(self, input_image, detections):
        im_height, im_width = input_image.shape[:-1]
        output_image = input_image
        for detection in detections['data']:
            label = detection['label']
            confidence = detection['confidence']
            bbox = detection['bounding_box']
            color = (254.0, 254.0, 254)
            output_image = cv2.rectangle(
                output_image,
                (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                color
            )
            label_conf = f'{label}: {confidence}'
            cv2.putText(output_image, label_conf, (bbox[0] - 10, bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output_image
