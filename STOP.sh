#!/bin/bash
# Stop all Healthcare Recommendation System services

echo "Stopping Healthcare Recommendation System..."

# Stop API
echo "  → Stopping FastAPI..."
pkill -9 -f 'uvicorn api.main'

# Stop Django
echo "  → Stopping Django..."
pkill -9 -f 'manage.py runserver'

sleep 1

echo ""
echo "✓ All services stopped"
echo ""
echo "Remaining processes:"
ps aux | grep -E "uvicorn|manage.py" | grep -v grep || echo "  (none)"
