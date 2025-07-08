.PHONY: up

up:
	docker compose down
	docker compose up

down:
	docker compose down

