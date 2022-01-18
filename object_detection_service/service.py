import functools
import threading
import uuid

from event_service_utils.logging.decorators import timer_logger
from event_service_utils.services.event_driven import BaseEventDrivenCMDService
from event_service_utils.tracing.jaeger import init_tracer


class ObjectDetectionService(BaseEventDrivenCMDService):
    def __init__(self,
                 service_stream_key, service_cmd_key_list,
                 pub_event_list, service_details,
                 file_storage_cli,
                 dnn_configs,
                 stream_factory,
                 logging_level,
                 tracer_configs):
        tracer = init_tracer(self.__class__.__name__, **tracer_configs)
        super(ObjectDetectionService, self).__init__(
            name=self.__class__.__name__,
            service_stream_key=service_stream_key,
            service_cmd_key_list=service_cmd_key_list,
            pub_event_list=pub_event_list,
            service_details=service_details,
            stream_factory=stream_factory,
            logging_level=logging_level,
            tracer=tracer,
        )
        self.cmd_validation_fields = ['id']
        self.data_validation_fields = ['id', 'image_url', 'data_flow', 'data_path', 'width', 'height', 'color_channels']

        self.fs_client = file_storage_cli
        self.dnn_configs = dnn_configs
        self.setup_model(self.dnn_configs)

    def setup_model(self, dnn_configs):
        # This package is the one inside tf_od_models folder
        if dnn_configs['model_type'] == 'tf_model_zoo':
            from tf_od_models.object_detection.coco_based_od import COCOBasedModel as Model
        elif dnn_configs['model_type'] == 'yolov3':
            from tf_od_models.yolo_ts_core.model_loader import YoloBasedModel as Model

        self.model = Model(base_configs=dnn_configs, lazy_setup=False)

    @timer_logger
    def process_data_event(self, event_data, json_msg):
        if not super(ObjectDetectionService, self).process_data_event(event_data, json_msg):
            return False
        model_result = self.extract_content(event_data)
        event_data = self.enrich_event_data(event_data, model_result)
        self.send_to_next_destinations(event_data)

    def process_action(self, action, event_data, json_msg):
        if not super(ObjectDetectionService, self).process_action(action, event_data, json_msg):
            return False

    @timer_logger
    def get_event_data_image_ndarray(self, event_data):
        img_key = event_data['image_url']
        width = event_data['width']
        height = event_data['height']
        color_channels = event_data['color_channels']
        n_channels = len(color_channels)
        nd_shape = (int(height), int(width), n_channels)
        image_nd_array = self.fs_client.get_image_ndarray_by_key_and_shape(img_key, nd_shape)

        return image_nd_array

    @timer_logger
    def extract_content(self, event_data):
        image_ndarray = self.get_event_data_image_ndarray(event_data)
        predictions = self.model.predict(image_ndarray)
        return predictions

    def node_tuple_from_obj_detection(self, obj_detection):
        node_tuples = ()
        for detection in obj_detection['data']:
            node_id = str(uuid.uuid4())
            node_attributes = {
                'id': node_id,
                'label': detection['label'],
                'confidence': detection['confidence'],
                'bounding_box': detection['bounding_box']
            }
            node = (
                node_id,
                node_attributes
            )
            node_tuples += (node,)
        return node_tuples

    def update_vekg(self, vekg, model_result):
        node_tuples = vekg.get('nodes', ())
        node_tuples += self.node_tuple_from_obj_detection(model_result)
        vekg['nodes'] = node_tuples
        return vekg

    @timer_logger
    def enrich_event_data(self, event_data, model_result):
        self.logger.debug('Enriching event data with model result')
        enriched_event_data = event_data.copy()

        enriched_event_data['vekg'] = self.update_vekg(enriched_event_data['vekg'], model_result)

        return enriched_event_data

    @functools.lru_cache(maxsize=5)
    def get_destination_streams(self, destination):
        return self.stream_factory.create(destination, stype='streamOnly')

    def send_event_to_destination(self, destination, event_data):
        self.logger.debug(f'Sending event to destination: {event_data} -> {destination}')
        destination_stream = self.get_destination_streams(destination)
        self.write_event_with_trace(event_data, destination_stream)

    def send_to_next_destinations(self, event_data):
        data_path = event_data.get('data_path', [])
        data_path.append(self.service_stream.key)
        next_data_flow_i = len(data_path)
        data_flow = event_data.get('data_flow', [])
        if next_data_flow_i >= len(data_flow):
            self.logger.info(f'Ignoring event without a next destination available: {event_data}')
            return

        next_destinations = data_flow[next_data_flow_i]
        for destination in next_destinations:
            self.send_event_to_destination(destination, event_data)

    def log_state(self):
        super(ObjectDetectionService, self).log_state()
        self._log_dict('DNN configs', self.dnn_configs)

    def run(self):
        self.log_state()
        super(ObjectDetectionService, self).run()
        self.data_thread = threading.Thread(target=self.run_forever, args=(self.process_data,))
        self.data_thread.start()
        self.data_thread.join()
