#!/usr/bin/env bash
export FLASK_APP=backend.app
exec flask run --reload --host 0.0.0.0 --port 5000 