# This file is preparation only; activation targets will be added once the
# model configuration, port allocation and ingress contract are established.
COMPOSE := podman-compose
COMPOSE_FILE := compose.yaml

.PHONY: help config-check
help:
	@printf '%s\n' 'config-check: validate a configuration with explicit port and bind address'

config-check:
	@test -n "$$OPENVIKING_BIND_ADDRESS" && test -n "$$OPENVIKING_HOST_PORT" || { printf '%s\n' 'Set OPENVIKING_BIND_ADDRESS and OPENVIKING_HOST_PORT first' >&2; exit 1; }
	$(COMPOSE) -f $(COMPOSE_FILE) config >/dev/null
