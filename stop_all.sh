#!/bin/bash
echo "Stopping services..."
pkill -f gateway-service
pkill -f vision-worker
pkill -f audio-worker
# Also kill hardware-controller if it was started in background
pkill -f hardware-controller
echo "All services stopped."
