"""
Console API: FastAPI REST commands + a WebSocket event stream over one ConsoleEngine.

Commands are synchronous endpoints (FastAPI runs them in its thread pool); the engine's lock
serializes them. Engine events (boot progress, strikes, diagnoses, repairs, mission steps, log
lines) are fanned out to every connected WebSocket. If frontend/dist exists it is served at "/",
so `python -m aegis.console` is a one-command demo.
"""

import asyncio
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aegis.console.engine import EngineBusy, EngineNotReady, ConsoleEngine
from aegis.console.mission import MissionConfig

FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "frontend", "dist")
LOG_BACKLOG = 300


def jsonable(x: Any) -> Any:
    """Recursively replaces non-finite floats (JSON has no NaN/Infinity)."""
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Infinity" if x > 0 else "-Infinity"
        return x
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


class EventHub:
    """Thread-safe fan-out of engine events to WebSocket subscribers on the server's event loop."""

    def __init__(self) -> None:
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.queues: Set[asyncio.Queue] = set()
        self.backlog: List[Dict[str, Any]] = []  # recent log lines for late joiners
        self.seq = 0
        self._lock = threading.Lock()

    def publish(self, kind: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.seq += 1
            event = {"type": kind, "seq": self.seq, "t": time.time(), "data": jsonable(payload)}
            if kind == "log":
                self.backlog.append(event)
                del self.backlog[:-LOG_BACKLOG]
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        for q in list(self.queues):
            loop.call_soon_threadsafe(self._offer, q, event)

    @staticmethod
    def _offer(q: asyncio.Queue, event: Dict[str, Any]) -> None:
        if q.full():  # a stalled client loses its oldest events rather than stalling the engine
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(event)


# ---- request bodies ------------------------------------------------------------------------

class StrikeBody(BaseModel):
    block: str = "random"
    field: str = "exponent_msb"
    count: int = Field(1, ge=1, le=64)
    param: Optional[str] = None


class ArmBody(BaseModel):
    armed: bool


class DiagnoseBody(BaseModel):
    full_crc: bool = False


class RepairBody(BaseModel):
    mode: str = "aegis"


class ClassifyBody(BaseModel):
    indices: List[int]


class EvaluateBody(BaseModel):
    scope: str = "probe"


class ArchBody(BaseModel):
    arch: str


class MissionBody(BaseModel):
    hours: float = 12.0
    step_minutes: float = 60.0
    flux: float = 10.0
    scrub_hours: float = 6.0
    audit_hours: float = 6.0
    probe_per_class: int = 2
    seed: int = 0


def create_app(arch: str = "resnet18", engine: Optional[ConsoleEngine] = None, boot: bool = True,
               serve_frontend: bool = True) -> FastAPI:
    hub = EventHub()
    engine = engine or ConsoleEngine()
    engine._emit_fn = hub.publish

    def boot_in_background(target_arch: str) -> None:
        def run():
            try:
                engine.boot(target_arch)
            except Exception:
                pass  # reported through engine status + events

        threading.Thread(target=run, name="engine-boot", daemon=True).start()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        if boot:
            boot_in_background(arch)
        yield
        if engine.mission_running:
            engine.stop_mission()

    app = FastAPI(title="Aegis-v2 Mission Console", lifespan=lifespan)
    app.state.engine = engine
    app.state.hub = hub

    def call(fn, *args, **kwargs):
        try:
            return JSONResponse(jsonable(fn(*args, **kwargs)))
        except EngineBusy as exc:
            raise HTTPException(409, str(exc))
        except EngineNotReady as exc:
            raise HTTPException(503, str(exc))
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/health")
    def health():
        return {"ok": True, "status": engine.status, "arch": engine.arch}

    @app.get("/api/state")
    def state():
        return call(engine.snapshot)

    @app.get("/api/gallery")
    def gallery():
        return call(engine.gallery)

    @app.get("/api/chip/{index}.png")
    def chip(index: int, scale: int = 1):
        try:
            png = engine.chip_png(index, scale=max(1, min(scale, 8)))
        except EngineNotReady as exc:
            raise HTTPException(503, str(exc))
        except ValueError as exc:
            raise HTTPException(404, str(exc))
        return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/field")
    def field():
        if engine.field is None:
            raise HTTPException(503, "Field not computed yet")
        return engine.field

    @app.post("/api/strike")
    def strike(body: StrikeBody):
        return call(engine.strike, body.block, body.field, body.count, body.param)

    @app.post("/api/arm")
    def arm(body: ArmBody):
        return call(engine.set_armed, body.armed)

    @app.post("/api/diagnose")
    def diagnose(body: DiagnoseBody):
        return call(engine.diagnose, body.full_crc)

    @app.post("/api/repair")
    def repair(body: RepairBody):
        return call(engine.repair, body.mode)

    @app.post("/api/reset")
    def reset():
        return call(engine.reset)

    @app.post("/api/classify")
    def classify(body: ClassifyBody):
        return call(engine.classify, body.indices)

    @app.post("/api/evaluate")
    def evaluate(body: EvaluateBody):
        return call(engine.evaluate, body.scope)

    @app.post("/api/arch")
    def switch_arch(body: ArchBody):
        if body.arch not in engine.available_archs():
            raise HTTPException(400, f"Architecture '{body.arch}' is not available")
        if engine.mission_running:
            raise HTTPException(409, "A live mission is running")
        if engine.status == "booting":
            raise HTTPException(409, "The engine is already booting")
        engine.status = "booting"
        boot_in_background(body.arch)
        return {"booting": body.arch}

    @app.post("/api/mission/start")
    def mission_start(body: MissionBody):
        return call(engine.start_mission, MissionConfig(**body.model_dump()))

    @app.post("/api/mission/stop")
    def mission_stop():
        return call(engine.stop_mission)

    @app.get("/api/mission")
    def mission():
        return call(lambda: engine.mission_state() or {})

    @app.websocket("/api/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        hub.queues.add(queue)
        try:
            await socket.send_json({"type": "hello", "seq": hub.seq, "t": time.time(),
                                    "data": jsonable({"state": engine.snapshot(), "logs": list(hub.backlog)})})
            while True:
                await socket.send_json(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.queues.discard(queue)

    if serve_frontend and os.path.isdir(FRONTEND_DIST):
        @app.middleware("http")
        async def revalidate_html(request, call_next):
            # index.html names content-hashed bundles; a cached copy after a rebuild would point at
            # deleted files, so pages always revalidate (the hashed assets themselves can cache)
            response = await call_next(request)
            if response.headers.get("content-type", "").startswith("text/html"):
                response.headers["Cache-Control"] = "no-cache"
            return response

        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    return app
