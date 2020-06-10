import os

from decouple import config

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SOURCE_DIR)

REDIS_ADDRESS = config('REDIS_ADDRESS', default='localhost')
REDIS_PORT = config('REDIS_PORT', default='6379')

TRACER_REPORTING_HOST = config('TRACER_REPORTING_HOST', default='localhost')
TRACER_REPORTING_PORT = config('TRACER_REPORTING_PORT', default='6831')

SERVICE_STREAM_KEY = config('SERVICE_STREAM_KEY')
SERVICE_CMD_KEY = config('SERVICE_CMD_KEY')

MODEL_NAME = config('MODEL_NAME', default='ssd_mobilenet_v1_coco_2017_11_17_rt')
INPUT_WIDTH = config('INPUT_WIDTH', default=300, cast=int)
INPUT_HEIGHT = config('INPUT_HEIGHT', default=300, cast=int)
DETECTION_THRESHOLD = config('DETECTION_THRESHOLD', default=0.5, cast=float)
ALLOW_MEMORY_GROWTH = config('ALLOW_MEMORY_GROWTH', default=True, cast=bool)
DNN_TF_GPU_FRACTION = config('DNN_TF_GPU_FRACTION', default=1.0, cast=float)

LOGGING_LEVEL = config('LOGGING_LEVEL', default='DEBUG')
