IMGNAME=am-convai

docker-build:
	docker build . -t $(IMGNAME)
