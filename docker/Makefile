TAG="funkey/division_detection:v0.1"

.PHONY: default push
default:
	-docker rmi -f $(TAG)
	docker build -t $(TAG) .

push: default
	docker push $(TAG)

test: default
	docker run --rm $(TAG)
