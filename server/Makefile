# Variables
IMAGE_NAME=sumobot_data_app
CONTAINER_NAME=sumobot_data_app
HOST_PORT=8501
CONTAINER_PORT=8501

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Refresh
setup:
	git fetch
	git pull

# Run the container with dynamic volume mounting
run:
	docker run --restart=always -d --network tunnel --name $(CONTAINER_NAME) -p $(HOST_PORT):$(CONTAINER_PORT) $(IMAGE_NAME)

# Stop and remove the container
clean:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	docker rmi $(IMAGE_NAME) || true

# Full process (setup, build, run)
all: clean setup build run