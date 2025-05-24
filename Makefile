.PHONY: up

up:
	docker compose down
	docker compose up --build

down:
	docker compose down

