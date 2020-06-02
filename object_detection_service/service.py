import threading

from event_service_utils.logging.decorators import timer_logger
from event_service_utils.services.tracer import BaseTracerService
from event_service_utils.tracing.jaeger import init_tracer

# This package is the one inside tf_od_models folder
from object_detection.coco_based_od import COCOBasedModel






def evaluate_coco(
        model_name='ssd_mobilenet_v1_coco_2017_11_17',
        tf_gpu_fraction=0.75):

    model = COCOBasedModel(model_name=model_name, tf_gpu_fraction=tf_gpu_fraction, lazy_setup=False)


    self.model.stop()



class ObjectDetectionService(BaseTracerService):
    def __init__(self,
                 service_stream_key, service_cmd_key,
                 stream_factory,
                 logging_level,
                 tracer_configs):
        tracer = init_tracer(self.__class__.__name__, **tracer_configs)
        super(ObjectDetectionService, self).__init__(
            name=self.__class__.__name__,
            service_stream_key=service_stream_key,
            service_cmd_key=service_cmd_key,
            stream_factory=stream_factory,
            logging_level=logging_level,
            tracer=tracer,
        )
        self.cmd_validation_fields = ['id', 'action']
        self.data_validation_fields = ['id']
        self.setup_model(model_name, tf_gpu_fraction)

    def setup_model(self, model_name, tf_gpu_fraction):
        self.model = COCOBasedModel(model_name=model_name, tf_gpu_fraction=tf_gpu_fraction, lazy_setup=False)

    # def send_event_to_somewhere(self, event_data):
    #     self.logger.debug(f'Sending event to somewhere: {event_data}')
    #     self.write_event_with_trace(event_data, self.somewhere_stream)

    @timer_logger
    def process_data_event(self, event_data, json_msg):
        if not super(ObjectDetectionService, self).process_data_event(event_data, json_msg):
            return False
        # do something here
        pass

    def process_action(self, action, event_data, json_msg):
        if not super(ObjectDetectionService, self).process_action(action, event_data, json_msg):
            return False
        if action == 'someAction':
            # do some action
            pass
        elif action == 'otherAction':
            # do some other action
            pass

    def log_state(self):
        super(ObjectDetectionService, self).log_state()
        self.logger.info(f'My service name is: {self.name}')

    def run(self):
        super(ObjectDetectionService, self).run()
        self.cmd_thread = threading.Thread(target=self.run_forever, args=(self.process_cmd,))
        self.data_thread = threading.Thread(target=self.run_forever, args=(self.process_data,))
        self.cmd_thread.start()
        self.data_thread.start()
        self.cmd_thread.join()
        self.data_thread.join()
