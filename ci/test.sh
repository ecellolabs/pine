#!/usr/bin/env bash

uv run coverage run --source=pine -m pytest $@
uv run coverage report --show-missing
