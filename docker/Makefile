TAG="funkey/division_detection:v0.2"

.PHONY: default push
default:
	-docker rmi -f $(TAG)
	docker pull funkey/gunpowder:v0.3
	docker build -t $(TAG) .

push: default
	docker push $(TAG)

test: default
	docker run --rm $(TAG)
