IMGNAME=am-convai

docker-run:
	docker run -p 1234:80 -t $(IMGNAME)

docker-build:
	docker build . -t $(IMGNAME)

