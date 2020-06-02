from setuptools import setup

setup(
    name='object_detection_service',
    version='0.1',
    description='This service is responsible for handling object detection on image streams and the output is a vekg of each image',
    author='Felipe Arruda Pontes',
    author_email='felipe.arruda.pontes@insight-centre.org',
    packages=['object_detection_service'],
    zip_safe=False
)
