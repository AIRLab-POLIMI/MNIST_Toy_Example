#!/bin/bash
readonly SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}")")
USERNAME=$(whoami)
IMAGE_NAME=$USERNAME/mnist_tutorial
TAG=dev
DOCKER_FILENAME=dockerfile


while [[ $# -gt 0 ]]
do key="$1"

case $key in
	-im|--image_name)
	IMAGE_NAME="$2"
	shift # past argument
	shift # past value
	;;
	-t|--tag)
	TAG="$2"
	shift # past argument
	shift # past value
	;;
	-h|--help)
	shift # past argument
	echo "Options:"
	echo "	-im, --image_name	name of the docker image (default \"base_images/tensorflow\")"
	echo "	-t, --tag		image tag name (default \"tf2.2-gpu\")"
	exit
	;;
	*)
	echo " Wrong option(s) is selected. Use -h, --help for more information "
	exit
	;;
esac
done

echo "${SCRIPT_DIR}/${DOCKER_FILENAME}"

docker build -t ${IMAGE_NAME}:${TAG} \
	-f ${SCRIPT_DIR}/${DOCKER_FILENAME} \
	${SCRIPT_DIR}
