.PHONY: all up down sparse crawler page

all:
	docker compose down
	docker compose up --build

up:
	docker compose up

down:
	docker compose down

sparse:
	docker compose down lookup video-crawler image-crawler
	docker compose up lookup video-crawler image-crawler --build

crawler:
	docker compose down video-crawler image-crawler
	docker compose up video-crawler image-crawler --build

ui:
	docker compose down lookup
	docker compose up lookup --build

page:
	docker compose down graph-page-worker analytics-worker
	docker compose up graph-page-worker analytics-worker --build

