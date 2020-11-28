import os
import zipfile

import six.moves.urllib as urllib

import numpy as np
import tensorflow as tf
import cv2


from tf_od_models.yolo_ts_core.utils.misc_utils import parse_anchors, read_class_names
from tf_od_models.yolo_ts_core.utils.nms_utils import gpu_nms
from tf_od_models.yolo_ts_core.utils.plot_utils import get_color_table, plot_one_box

from tf_od_models.yolo_ts_core.model import yolov3

from tf_od_models.object_detection.coco_based_od import COCOBasedModel


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def download_model(model_name=None, model_path=None):
    model_file_name = model_name + '.zip'
    model_file_path = os.path.join(CURRENT_DIR, 'models', model_file_name)

    if model_path is None:
        model_path = os.path.join(CURRENT_DIR, 'models', f'{model_name}')
    model_meta = model_path + '.meta'
    if not os.path.isfile(model_meta):
        url = f"https://www.dropbox.com/s/aylf4ng9smqblw8/{model_file_name}?dl=1"
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
    return model_path


def get_session_and_model_ready(model_name, model_path,
                                width=416, height=416,
                                detection_threshold=0.5,
                                allow_memory_growth=True, tf_gpu_fraction=1.0):

    class_name_path = model_path + '.names'
    classes = read_class_names(class_name_path)
    num_class = len(classes)
    category_index = {c_id: {'name': label} for c_id, label in classes.items()}
    color_table = get_color_table(num_class)

    anchor_path = model_path + '.anchors'
    anchors = parse_anchors(anchor_path)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=tf_gpu_fraction, allow_growth=allow_memory_growth)
#
    # detection_graph, category_index = load_frozenmodel(model_name=model_name)
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=gpu_options)
    )
    sess.__enter__()
    tf.global_variables_initializer().run(session=sess)

    input_data = tf.placeholder(tf.float32, [1, height, width, 3], name='input_data')
    yolo_model = yolov3(num_class, anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)

    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(
        pred_boxes, pred_scores, num_class, max_boxes=30, score_thresh=detection_threshold, iou_thresh=0.5)

    saver = tf.train.Saver()
    print(model_path)
    saver.restore(sess, model_path)

    model_configs = {
        'session': sess,
        'category_index': category_index,
        'detection_threshold': detection_threshold,
        'image_tensor': input_data,
        'detection_boxes': boxes,
        'detection_scores': scores,
        'detection_classes': labels,
        'color_table': color_table,
        # 'classes': classes,
    }
    return model_configs


def predict(model_configs, preprocessed_image):
    detection_boxes = model_configs.get('detection_boxes')
    detection_scores = model_configs.get('detection_scores')
    detection_classes = model_configs.get('detection_classes')
    image_tensor = model_configs.get('image_tensor')

    sess = model_configs.get('session')
    (boxes, scores, classes) = sess.run(
        [detection_boxes, detection_scores, detection_classes],
        feed_dict={image_tensor: preprocessed_image}
    )
    return (boxes, scores, classes, None)


def resize_image(input_image, input_height, input_width):
    resized_image = cv2.resize(
        input_image, (input_height, input_width), interpolation=cv2.INTER_CUBIC)
    return resized_image


def preprocessing(input_image):
    color_changed_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    image_array = np.asarray(color_changed_image, np.float32)
    image_array = image_array[np.newaxis, :] / 255.
    return image_array


def post_processing(boxes, scores, classes, detection_threshold,
                    category_index, origin_height, origin_width, new_height, new_width):
    output = []
    boxes = normalized_bbox(boxes, origin_height, origin_width, new_height, new_width)
    for bbox, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
        if score > detection_threshold:
            label = category_index[_class]['name']
            obj = {
                'label': label,
                'bounding_box': [int(i) for i in bbox],
                'confidence': float(score)
            }
            output.append(obj)
    return {'data': output}


def normalized_bbox(bboxes, origin_height, origin_width, new_height, new_width):

    # rescale the coordinates to the original image
    bboxes[:, 0] *= (origin_width / float(new_width))
    bboxes[:, 2] *= (origin_width / float(new_width))
    bboxes[:, 1] *= (origin_height / float(new_height))
    bboxes[:, 3] *= (origin_height / float(new_height))

    return bboxes


class YoloBasedModel(COCOBasedModel):
    # def __init__(self, base_configs, lazy_setup=False):
    #     super(COCOHatBasedModel, self).__init__()
    #     self.base_configs = base_configs
    #     if not lazy_setup:
    #         self.setup()

    def setup(self):
        model_name = self.base_configs['model_name']
        width = self.base_configs['width']
        height = self.base_configs['height']
        detection_threshold = self.base_configs['detection_threshold']
        allow_memory_growth = self.base_configs['allow_memory_growth']
        tf_gpu_fraction = self.base_configs['tf_gpu_fraction']
        model_path = download_model(model_name=model_name)
        self.model_configs = get_session_and_model_ready(
            model_name=model_name, model_path=model_path,
            width=width, height=height,
            detection_threshold=detection_threshold,
            allow_memory_growth=allow_memory_growth, tf_gpu_fraction=tf_gpu_fraction)

        if self.base_configs['hot_start'] is True:
            print('Running hot start...')
            self._hot_start(width, height)
            print('Finished hot start...')

    # def to_debug(self, boxes_, scores_, labels_, classes, color_table, width_ori, height_ori, new_size=(416, 416)):

    #     # rescale the coordinates to the original image
    #     boxes_[:, 0] *= (width_ori / float(new_size[0]))
    #     boxes_[:, 2] *= (width_ori / float(new_size[0]))
    #     boxes_[:, 1] *= (height_ori / float(new_size[1]))
    #     boxes_[:, 3] *= (height_ori / float(new_size[1]))

    #     print("box coords:")
    #     print(boxes_)
    #     print('*' * 30)
    #     print("scores:")
    #     print(scores_)
    #     print('*' * 30)
    #     print("labels:")
    #     print(labels_)

    #     for i in range(len(boxes_)):
    #         x0, y0, x1, y1 = boxes_[i]
    #         plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]], color=color_table[labels_[i]])
    #     # cv2.imshow('Detection result', img_ori)
    #     image_new_path = 'result.jpg'
    #     cv2.imwrite(image_new_path, img_ori)

    def predict(self, input_image):
        origin_height, origin_width = input_image.shape[:-1]
        category_index = self.model_configs['category_index']
        detection_threshold = self.model_configs['detection_threshold']
        width = self.base_configs['width']
        height = self.base_configs['height']
        resized_image = resize_image(input_image, height, width)
        preprocessed_image = preprocessing(resized_image)
        (boxes, scores, classes, num) = predict(self.model_configs, preprocessed_image)

        # self.to_debug(boxes, scores, labels_, classes, color_table, origin_width, origin_height, new_size=(416, 416))

        detections = post_processing(
            boxes, scores, classes, detection_threshold, category_index,
            origin_height, origin_width,
            height, width
        )
        return detections


def read_image_from_path(image_path):
    img_ori = cv2.imread(image_path)
    return img_ori


if __name__ == '__main__':
    model = YoloBasedModel(
        base_configs={
            'model_name': 'hat_not_hat',
            'width': 416,
            'height': 416,
            'detection_threshold': 0.45,
            'allow_memory_growth': True,
            'tf_gpu_fraction': 0.75,
            'hot_start': True,
        }
    )
    print('ok')
    img_ori = read_image_from_path('./data/demo_data/pictor1.jpg')
    detections = model.predict(img_ori)
    print(detections)
    bb_img = model.add_bbboxes_to_image(img_ori, detections)

    image_new_path = 'result.jpg'
    cv2.imwrite(image_new_path, bb_img)
    model.stop()
