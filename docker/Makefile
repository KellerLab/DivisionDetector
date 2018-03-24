TAG="funkey/division_detection:v0.1"

.PHONY: default push
default:
	-docker rmi -f $(TAG)
	docker pull funkey/gunpowder:v0.3-pre8
	docker build -t $(TAG) .

push: default
	docker push $(TAG)

test: default
	docker run --rm $(TAG)
