from setuptools import setup

setup(
    name='object_detection',
    version='0.1',
    description='Necessary to make work with underlying requirements from the solution extracted from TF object model zoo. some label-to-int or whatever resolution is required, which is also protobuff based, which makes it necessary to put it in here.',
    author='Felipe Arruda Pontes',
    author_email='felipe.arruda.pontes@insight-centre.org',
    packages=['object_detection'],
    zip_safe=False
)
