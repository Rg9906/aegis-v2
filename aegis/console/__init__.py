"""
Live Mission Console.

engine.py  - stateful, thread-safe operator session over the real Aegis-v2 pipeline
mission.py - live orbit missions (identical fault stream applied to strategy replicas)
server.py  - FastAPI REST + WebSocket API consumed by the React console (frontend/)

Run with `python -m aegis.console`.
"""
