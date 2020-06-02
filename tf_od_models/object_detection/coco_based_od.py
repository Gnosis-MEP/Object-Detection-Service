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
        model_name = 'ssd_mobilenet_v11_coco'
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


def preprocess(model_configs, image_np):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    return image_np_expanded


def predict(model_configs, image_path):
    detection_boxes = model_configs.get('detection_boxes')
    detection_scores = model_configs.get('detection_scores')
    detection_classes = model_configs.get('detection_classes')
    num_detections = model_configs.get('num_detections')
    image_tensor = model_configs.get('image_tensor')
    category_index = model_configs.get('category_index')
    detection_threshold = model_configs.get('detection_threshold')

    sess = model_configs.get('session')

    image_np = cv2.imread(image_path, 1)
    preproc_image_np = preprocess(model_configs, image_np)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: preproc_image_np}
    )
    return json_serializable_output_prediction(boxes, scores, classes, detection_threshold, category_index)


def json_serializable_output_prediction(boxes, scores, classes, detection_threshold, category_index):
    output = []
    for box, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
        if score > detection_threshold:
            label = category_index[_class]['name']
            obj = {
                'label': label,
                'bounding_box': [float(value) for value in box],
                'confidence': float(score)
            }
            output.append(obj)
    return output


class COCOBasedModel(object):
    def __init__(self, model_name, tf_gpu_fraction=0.75, lazy_setup=False):
        super(COCOBasedModel, self).__init__()
        self.tf_gpu_fraction = tf_gpu_fraction
        self.model_name = model_name
        if not lazy_setup:
            self.setup()

    def setup(self):
        download_model(model_name=self.model_name)
        self.model_configs = get_session_and_model_ready(
            model_name=self.model_name,
            tf_gpu_fraction=self.tf_gpu_fraction
        )

    def predict(self, image_path):
        return predict(self.model_configs, image_path)

    def stop(self):
        self.model_configs['session'].close()
