# Object Detection Service
This service is responsible for handling object detection on image streams and the output is a vekg of each image.

It should be capable of running any Tensorflow 1.12-1.14 object detection trained on COCO dataset from the [model zoo](https://github.com/tensorflow/models/blob/r1.12.0/research/object_detection/g3doc/detection_model_zoo.md) (The ones that output masks were not tested.)


# Events Listened
 - [VEKG](https://github.com/Gnosis-MEP/Gnosis-Docs/blob/main/EventTypes.md#VEKG)

# Events Published
 - [SERVICE_WORKER_ANNOUNCED](https://github.com/Gnosis-MEP/Gnosis-Docs/blob/main/EventTypes.md#SERVICE_WORKER_ANNOUNCED)
 - [VEKG](https://github.com/Gnosis-MEP/Gnosis-Docs/blob/main/EventTypes.md#VEKG)


# Installation

## Configure .env
Copy the `example.env` file to `.env`, and inside it replace the variables with the values you need.

## Installing Dependencies

### Using pipenv
Run `$ pipenv shell` to create a python virtualenv and load the .env into the environment variables in the shell.

Then run: `$ pipenv install` to install all packages, or `$ pipenv install -d` to also install the packages that help during development, eg: ipython.
This runs the installation using **pip** under the hood, but also handle the cross dependency issues between packages and checks the packages MD5s for security mesure.


### Using pip
To install from the `requirements.txt` file, run the following command:
```
$ pip install -r requirements.txt
```

# Running
Enter project python environment (virtualenv or conda environment)

**ps**: It's required to have the .env variables loaded into the shell so that the project can run properly. An easy way of doing this is using `pipenv shell` to start the python environment with the `.env` file loaded or using the `source load_env.sh` command inside your preferable python environment (eg: conda).

Then, run the service with:
```
$ ./object_detection_service/run.py
```

# Testing
Run the script `run_tests.sh`, it will run all tests defined in the **tests** directory.

Also, there's a python script at `./object_detection_service/send_msgs_test.py` to do some simple manual testing, by sending msgs to the service stream key.


# Docker
## Build
Build the docker image using: `docker-compose build`

**ps**: It's required to have the .env variables loaded into the shell so that the container can build properly. An easy way of doing this is using `pipenv shell` to start the python environment with the `.env` file loaded or using the `source load_env.sh` command inside your preferable python environment (eg: conda).

## Run
Use `docker-compose run --rm service` to run the docker image

## Benchmark Tests
To run the benchmark tests one needs to manually start the Benchmark stage in the CI pipeline (Gitlab), it shoud be enabled after the tests stage is done. Only by passing the benchmark tests shoud the image be tagged with 'latest', to show that it is a stable docker image.
