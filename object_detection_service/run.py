#!/usr/bin/env python
from event_service_utils.streams.redis import RedisStreamFactory
from event_service_utils.img_serialization.redis import RedisImageCache

from object_detection_service.service import ObjectDetectionService

from object_detection_service.conf import (
    REDIS_ADDRESS,
    REDIS_PORT,
    REDIS_MAX_STREAM_SIZE,
    SERVICE_STREAM_KEY,
    SERVICE_CMD_KEY,
    LOGGING_LEVEL,
    TRACER_REPORTING_HOST,
    TRACER_REPORTING_PORT,
    MODEL_NAME,
    MODEL_TYPE,
    INPUT_WIDTH,
    INPUT_HEIGHT,
    DETECTION_THRESHOLD,
    ALLOW_MEMORY_GROWTH,
    DNN_TF_GPU_FRACTION,
    DNN_HOT_START,
)


def run_service():
    tracer_configs = {
        'reporting_host': TRACER_REPORTING_HOST,
        'reporting_port': TRACER_REPORTING_PORT,
    }

    redis_fs_cli_config = {
        'host': REDIS_ADDRESS,
        'port': REDIS_PORT,
        'db': 0,
    }

    file_storage_cli = RedisImageCache()
    file_storage_cli.file_storage_cli_config = redis_fs_cli_config
    file_storage_cli.initialize_file_storage_client()

    dnn_configs = {
        'model_name': MODEL_NAME,
        'width': INPUT_WIDTH,
        'height': INPUT_HEIGHT,
        'detection_threshold': DETECTION_THRESHOLD,
        'allow_memory_growth': ALLOW_MEMORY_GROWTH,
        'tf_gpu_fraction': DNN_TF_GPU_FRACTION,
        'hot_start': DNN_HOT_START,
        'model_type': MODEL_TYPE,
    }

    stream_factory = RedisStreamFactory(host=REDIS_ADDRESS, port=REDIS_PORT, max_stream_length=REDIS_MAX_STREAM_SIZE)

    service = ObjectDetectionService(
        service_stream_key=SERVICE_STREAM_KEY,
        service_cmd_key=SERVICE_CMD_KEY,
        file_storage_cli=file_storage_cli,
        dnn_configs=dnn_configs,
        stream_factory=stream_factory,
        logging_level=LOGGING_LEVEL,
        tracer_configs=tracer_configs
    )
    service.run()


def main():
    try:
        run_service()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
