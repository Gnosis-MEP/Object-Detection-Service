import os

from decouple import config, Csv


def int_or_none(val):
    if val is None:
        return None
    int_val = int(val)
    if int_val > 0:
        return int_val
    return None


SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SOURCE_DIR)

REDIS_ADDRESS = config('REDIS_ADDRESS', default='localhost')
REDIS_PORT = config('REDIS_PORT', default='6379')

REDIS_MAX_STREAM_SIZE = config('REDIS_MAX_STREAM_SIZE', default=None, cast=int_or_none)

TRACER_REPORTING_HOST = config('TRACER_REPORTING_HOST', default='localhost')
TRACER_REPORTING_PORT = config('TRACER_REPORTING_PORT', default='6831')

SERVICE_STREAM_KEY = config('SERVICE_STREAM_KEY')
SERVICE_CMD_KEY = config('SERVICE_CMD_KEY')
SERVICE_REGISTRY_CMD_KEY = config('SERVICE_REGISTRY_CMD_KEY')

SERVICE_DETAILS_SERVICE_TYPE = config('SERVICE_DETAILS_SERVICE_TYPE')
SERVICE_DETAILS_STREAM_KEY = config('SERVICE_DETAILS_STREAM_KEY')
SERVICE_DETAILS_QUEUE_LIMIT = config('SERVICE_DETAILS_QUEUE_LIMIT', cast=int)
SERVICE_DETAILS_THROUGHPUT = config('SERVICE_DETAILS_THROUGHPUT', cast=float)
SERVICE_DETAILS_ACCURACY = config('SERVICE_DETAILS_ACCURACY', cast=float)
SERVICE_DETAILS_ENERGY_CONSUMPTION = config('SERVICE_DETAILS_ENERGY_CONSUMPTION', cast=float)
SERVICE_DETAILS_CONTENT_TYPES = config('SERVICE_DETAILS_CONTENT_TYPES', cast=Csv())


SERVICE_DETAILS = {
    'service_type': SERVICE_DETAILS_SERVICE_TYPE,
    'stream_key': SERVICE_DETAILS_STREAM_KEY,
    'queue_limit': SERVICE_DETAILS_QUEUE_LIMIT,
    'throughput': SERVICE_DETAILS_THROUGHPUT,
    'accuracy': SERVICE_DETAILS_ACCURACY,
    'energy_consumption': SERVICE_DETAILS_ENERGY_CONSUMPTION,
    'content_types': SERVICE_DETAILS_CONTENT_TYPES
}

SERVICE_CMD_KEY_LIST = []

PUB_EVENT_TYPE_SERVICE_WORKER_ANNOUNCED = config('PUB_EVENT_TYPE_SERVICE_WORKER_ANNOUNCED')

PUB_EVENT_LIST = [
    PUB_EVENT_TYPE_SERVICE_WORKER_ANNOUNCED,
]


MODEL_NAME = config('MODEL_NAME', default='ssd_mobilenet_v1_coco_2017_11_17_rt')
MODEL_TYPE = config('MODEL_TYPE', default='tf_model_zoo')
INPUT_WIDTH = config('INPUT_WIDTH', default=300, cast=int)
INPUT_HEIGHT = config('INPUT_HEIGHT', default=300, cast=int)
DETECTION_THRESHOLD = config('DETECTION_THRESHOLD', default=0.5, cast=float)
ALLOW_MEMORY_GROWTH = config('ALLOW_MEMORY_GROWTH', default=True, cast=bool)
DNN_TF_GPU_FRACTION = config('DNN_TF_GPU_FRACTION', default=1.0, cast=float)
DNN_HOT_START = config('DNN_HOT_START', default=True, cast=bool)

LOGGING_LEVEL = config('LOGGING_LEVEL', default='DEBUG')
