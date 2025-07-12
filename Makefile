.PHONY: up down sparse crawler

up:
	docker compose down
	docker compose up --build

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

