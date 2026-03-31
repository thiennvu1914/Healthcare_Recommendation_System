#!/bin/bash
# Quick Start Script for Healthcare Recommendation System

echo "=========================================="
echo "Healthcare Recommendation System - Startup"
echo "=========================================="
echo ""

# Navigate to project directory
cd /root/Healthcare_Recommendation_System

# Check if data exists
if [ ! -f "data/articles.csv" ] || [ ! -f "data/QAs.csv" ]; then
    echo "⚠️  Warning: Data files not found in data/"
    echo "   Please ensure articles.csv and QAs.csv are present"
    exit 1
fi

echo "✓ Data files found"

# Start API Backend
echo ""
echo "Starting FastAPI Backend (port 8000)..."
SAMPLE_SIZE=0 ENABLE_CACHE=1 ENABLE_LLM_GENERATION=1 nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
API_PID=$!
echo "  → API PID: $API_PID"
echo "  → Logs: tail -f api.log"

# Wait for API to be ready
echo ""
echo "Waiting for API to start..."
for i in {1..60}; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "  ✓ API ready!"
        break
    fi
    echo -n "."
    sleep 1
done

# Start Django Web
echo ""
echo "Starting Django Web (port 8080)..."
cd web
nohup python manage.py runserver 0.0.0.0:8080 > web.log 2>&1 &
WEB_PID=$!
echo "  → Web PID: $WEB_PID"
echo "  → Logs: tail -f web/web.log"
cd ..

sleep 3

echo ""
echo "=========================================="
echo "✓ System started successfully!"
echo "=========================================="
echo ""
echo "Access URLs:"
echo "  • Web UI:  http://localhost:8080"
echo "  • AI Chat: http://localhost:8080/ai-advisor/"
echo "  • API:     http://localhost:8000/docs"
echo ""
echo "Process IDs:"
echo "  • API:     $API_PID"
echo "  • Web:     $WEB_PID"
echo ""
echo "To stop all services:"
echo "  pkill -f 'uvicorn api.main'"
echo "  pkill -f 'manage.py runserver'"
echo ""
echo "To view logs:"
echo "  tail -f api.log"
echo "  tail -f web/web.log"
echo ""
