# Runtime lifecycle for this project. Credentials are supplied separately.
COMPOSE := podman-compose -p openviking
COMPOSE_FILE := compose.yaml
.DEFAULT_GOAL := help

.PHONY: help config-check up stop status logs compose-up-help
compose-up-help:
	$(COMPOSE) -f $(COMPOSE_FILE) up --help

up:
	$(COMPOSE) -f $(COMPOSE_FILE) up -d --no-build --no-recreate

stop:
	$(COMPOSE) -f $(COMPOSE_FILE) stop

status:
	$(COMPOSE) -f $(COMPOSE_FILE) ps

logs:
	$(COMPOSE) -f $(COMPOSE_FILE) logs --tail=100

help:
	@printf '%s\n' 'config-check: validate a configuration with explicit port and bind address'

config-check:
	@test -n "$$OPENVIKING_BIND_ADDRESS" && test -n "$$OPENVIKING_HOST_PORT" || { printf '%s\n' 'Set OPENVIKING_BIND_ADDRESS and OPENVIKING_HOST_PORT first' >&2; exit 1; }
	$(COMPOSE) -f $(COMPOSE_FILE) config >/dev/null
