import json
import time
import os
import asyncio
import textwrap
import urllib.error
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load env file
load_dotenv()

from app.database import (
    SessionLocal,
    BlockModel,
    AgentModel,
    AgentMemoryModel,
    init_db,
    seed_if_empty
)

# Global variables
main_loop = None
WORLD_STARTED_AT = time.time()

GLOBAL_CHATS = [
    {
        "id": 1,
        "agent_id": "sim-bob",
        "agent_name": "BuilderBob",
        "message": "Hello world! Just logged into the voxel world.",
        "timestamp": time.time() - 300
    },
    {
        "id": 2,
        "agent_id": "sim-alice",
        "agent_name": "AliceVoxel",
        "message": "Hey BuilderBob! I'm planning to build a garden at x=5, z=5.",
        "timestamp": time.time() - 250
    }
]
background_tasks = set()

async def simulate_global_chat_loop():
    import random
    simulated_agents = [
        {"id": "sim-bob", "name": "BuilderBob"},
        {"id": "sim-alice", "name": "AliceVoxel"},
        {"id": "sim-master", "name": "VoxelMaster"},
        {"id": "sim-lego", "name": "LegoLover"},
        {"id": "sim-block", "name": "Blocky"}
    ]
    chat_pool = [
        "Just placed a block of glass at the top of my tower. The view is amazing!",
        "Has anyone tried building a castle at coordinates (20, 1, -15)?",
        "I need more wood blocks. Running low.",
        "Who is placing water blocks everywhere? It's flooding!",
        "My house is finally complete. 5x5 brick wall with a stone roof.",
        "I'm exploring the north side of the map.",
        "Ollama is running a bit slow today, but my builder is doing great!",
        "Is anyone online? I want to show my brick path.",
        "Just spawned a huge forest near spawn. Green everywhere!",
        "Making a glass dome. It takes so many ticks!",
        "Trying to build a bridge across the water stream.",
        "Hey! Let's collaborate on a castle.",
        "I'm setting my memory to remember my home coordinates.",
        "My goal today is to build a massive pyramid.",
        "Just finished clearing some dirt blocks."
    ]
    while True:
        try:
            await asyncio.sleep(random.randint(20, 40))
            agent = random.choice(simulated_agents)
            message = random.choice(chat_pool)
            new_msg = {
                "id": len(GLOBAL_CHATS) + 1,
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "message": message,
                "timestamp": time.time()
            }
            GLOBAL_CHATS.append(new_msg)
            broadcast_sync({
                "event": "global_chat",
                "message": new_msg
            })
        except asyncio.CancelledError:
            break
        except Exception as e:
            print("Error in simulate_global_chat_loop:", e)
            await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_loop
    main_loop = asyncio.get_running_loop()
    init_db()
    seed_if_empty()
    
    # Simulated chat background task is frozen/disabled to focus on a single agent.
    # task = asyncio.create_task(simulate_global_chat_loop())
    # background_tasks.add(task)
    # task.add_done_callback(background_tasks.discard)
    
    yield

app = FastAPI(title="Voxel World API", version="0.1.0", lifespan=lifespan)

# CORS setup
ENV = os.getenv("ENV", "development")
ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "")

if ENV == "production":
    if ALLOWED_ORIGINS_ENV:
        origins = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()]
    else:
        origins = []
else:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # No request in this app relies on cookies/credentials (auth is via request
    # body API keys), so keep this False - combining it with a wildcard origin
    # is an invalid combination that browsers reject for credentialed requests.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# WebSocket Connection Manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

manager = ConnectionManager()

def broadcast_sync(message: dict):
    if main_loop is not None:
        asyncio.run_coroutine_threadsafe(manager.broadcast(message), main_loop)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive data to keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

# ---------------------------------------------------------------------------
# World State Configuration
# ---------------------------------------------------------------------------
BLOCK_TYPES = {"grass", "dirt", "stone", "wood", "leaves", "water", "sand", "glass", "brick"}
DEFAULT_AGENT_ID = "web-builder"
DEFAULT_AGENT_NAME = "Builder"
DEFAULT_AGENT_COLOR = "#f5d000"
MAX_TOOL_CALLS_PER_TICK = 8

# Helper for memory storage
def get_agent_memory_dict(db, agent_id: str) -> dict[str, str]:
    rows = db.query(AgentMemoryModel).filter_by(agent_id=agent_id).all()
    return {row.key: row.value for row in rows}

def set_agent_memory(db, agent_id: str, key: str, value: str):
    row = db.query(AgentMemoryModel).filter_by(agent_id=agent_id, key=key).first()
    if row:
        row.value = value
    else:
        row = AgentMemoryModel(agent_id=agent_id, key=key, value=value)
        db.add(row)
    db.commit()

def get_agent_memory(db, agent_id: str, key: str) -> str | None:
    row = db.query(AgentMemoryModel).filter_by(agent_id=agent_id, key=key).first()
    return row.value if row else None

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Block(BaseModel):
    x: int
    y: int
    z: int
    type: str = Field(..., description="Block type, e.g. 'grass', 'stone', 'wood'")

class BlockDelete(BaseModel):
    x: int
    y: int
    z: int

class StructureSpawn(BaseModel):
    name: str  # "small_house" | "tree" | "wall"
    x: int
    y: int
    z: int
    rotation: int = 0  # 0/90/180/270 degrees, applied around y-axis

class AgentRegister(BaseModel):
    agent_id: str
    name: str
    x: int = 0
    y: int = 1
    z: int = 0
    color: str = "#f5d000"

class AgentMove(BaseModel):
    x: int
    y: int
    z: int

# Old agent run models for compatibility
class AgentRunRequest(BaseModel):
    provider: str = Field(..., description="groq, openai, gemini, or openai-compatible")
    model: str | None = None
    api_key: str
    goal: str
    base_url: str | None = None

class AgentRunResponse(BaseModel):
    ok: bool
    tick: int
    agent: dict
    thought: str = ""
    actions: list[dict]
    block_count: int

# New agent run models (Task 3)
class AgentRunRequestBody(BaseModel):
    provider: str = Field(..., description="gemini, openai, anthropic, groq, or ollama")
    model: str = Field(..., description="Model identifier")
    api_key: str = Field(..., description="API Key for the provider")
    goal: str = Field(..., description="Agent goal")
    ticks: int = Field(1, ge=1, le=3, description="Number of ticks to run (1-3)")

class TickSummary(BaseModel):
    tick: int
    thought: str
    actions: list[dict]

class AgentRunSummaryResponse(BaseModel):
    ok: bool
    agent_id: str
    ticks_run: int
    agent: dict
    ticks_detail: list[TickSummary]

# ---------------------------------------------------------------------------
# World read/write endpoints
# ---------------------------------------------------------------------------
@app.get("/world")
def get_world():
    """Full world snapshot. Fine at prototype scale; chunk this later."""
    db = SessionLocal()
    try:
        db_blocks = db.query(BlockModel).all()
        db_agents = db.query(AgentModel).all()
        return {
            "blocks": [{"x": b.x, "y": b.y, "z": b.z, "type": b.type} for b in db_blocks],
            "agents": [{"agent_id": a.agent_id, "name": a.name, "x": a.x, "y": a.y, "z": a.z, "color": a.color} for a in db_agents],
            "block_count": len(db_blocks),
            "uptime_seconds": round(time.time() - WORLD_STARTED_AT, 1),
        }
    finally:
        db.close()

@app.get("/world/chunk")
def get_chunk(x: int, z: int, radius: int = 16, y_min: int = -4, y_max: int = 20):
    """Localized view, e.g. what an agent standing at (x, *, z) can 'see'."""
    db = SessionLocal()
    try:
        db_blocks = db.query(BlockModel).filter(
            BlockModel.x >= x - radius,
            BlockModel.x <= x + radius,
            BlockModel.z >= z - radius,
            BlockModel.z <= z + radius,
            BlockModel.y >= y_min,
            BlockModel.y <= y_max
        ).all()
        return {"blocks": [{"x": b.x, "y": b.y, "z": b.z, "type": b.type} for b in db_blocks]}
    finally:
        db.close()

@app.post("/block")
def place_block(block: Block):
    if block.type not in BLOCK_TYPES:
        raise HTTPException(400, f"Unknown block type '{block.type}'. Valid types: {sorted(BLOCK_TYPES)}")
    db = SessionLocal()
    try:
        res = _execute_agent_tool(db, "place_block", {"x": block.x, "y": block.y, "z": block.z, "type": block.type}, "")
        if "error" in res:
            raise HTTPException(400, res["error"])
        return {"ok": True, "block": block}
    finally:
        db.close()

@app.delete("/block")
def remove_block(block: BlockDelete):
    db = SessionLocal()
    try:
        res = _execute_agent_tool(db, "remove_block", {"x": block.x, "y": block.y, "z": block.z}, "")
        if "error" in res:
            raise HTTPException(404, res["error"])
        return {"ok": True}
    finally:
        db.close()

# ---------------------------------------------------------------------------
# Prefab Structures
# ---------------------------------------------------------------------------
def _rotate(dx: int, dz: int, rotation: int) -> tuple[int, int]:
    rotation = rotation % 360
    if rotation == 0:
        return dx, dz
    if rotation == 90:
        return -dz, dx
    if rotation == 180:
        return -dx, -dz
    if rotation == 270:
        return dz, -dx
    raise HTTPException(400, "rotation must be one of 0, 90, 180, 270")

def _structure_small_house(ox: int, oy: int, oz: int, rotation: int):
    blocks = []
    w, d, h = 5, 5, 3
    for x in range(w):
        for z in range(d):
            blocks.append((x, 0, z, "wood"))
    for y in range(1, h):
        for x in range(w):
            for z in (0, d - 1):
                if y == 1 and x == 2 and z == 0:
                    continue
                blocks.append((x, y, z, "brick"))
        for z in range(d):
            for x in (0, w - 1):
                blocks.append((x, y, z, "brick"))
    blocks = [b for b in blocks if not (b[0] in (0, w - 1) and b[2] not in (0, d - 1) and b[1] >= 1)]
    for y in range(1, h):
        for z in range(d):
            for x in (0, w - 1):
                blocks.append((x, y, z, "brick"))
    blocks = [b for b in blocks if not (b[1] == 1 and b[0] in (0, w - 1) and b[2] == 2)]
    for z in (2,):
        for x in (0, w - 1):
            blocks.append((x, 1, z, "glass"))
    for x in range(-1, w + 1):
        for z in range(-1, d + 1):
            blocks.append((x, h, z, "stone"))

    out = []
    for dx, dy, dz, t in blocks:
        rx, rz = _rotate(dx, dz, rotation)
        out.append((ox + rx, oy + dy, oz + rz, t))
    return out

def _structure_tree(ox: int, oy: int, oz: int, rotation: int):
    blocks = []
    trunk_h = 4
    for y in range(trunk_h):
        blocks.append((0, y, 0, "wood"))
    for dy in range(2, 5):
        r = 2 if dy < 4 else 1
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if dx * dx + dz * dz <= r * r + 1:
                    blocks.append((dx, trunk_h - 1 + dy - 1, dz, "leaves"))
    out = []
    for dx, dy, dz, t in blocks:
        rx, rz = _rotate(dx, dz, rotation)
        out.append((ox + rx, oy + dy, oz + rz, t))
    return out

def _structure_wall(ox: int, oy: int, oz: int, rotation: int, length: int = 6, height: int = 2):
    blocks = []
    for x in range(length):
        for y in range(height):
            blocks.append((x, y, 0, "stone"))
    out = []
    for dx, dy, dz, t in blocks:
        rx, rz = _rotate(dx, dz, rotation)
        out.append((ox + rx, oy + dy, oz + rz, t))
    return out

STRUCTURES = {
    "small_house": _structure_small_house,
    "tree": _structure_tree,
    "wall": _structure_wall,
}

@app.get("/structures")
def list_structures():
    return {"available": list(STRUCTURES.keys())}

@app.post("/structure")
def spawn_structure(s: StructureSpawn):
    if s.name not in STRUCTURES:
        raise HTTPException(400, f"Unknown structure '{s.name}'. Available: {list(STRUCTURES.keys())}")
    db = SessionLocal()
    try:
        res = _execute_agent_tool(db, "spawn_structure", {"name": s.name, "x": s.x, "y": s.y, "z": s.z, "rotation": s.rotation}, "")
        if "error" in res:
            raise HTTPException(400, res["error"])
        return {"ok": True, "blocks_placed": res["blocks_placed"]}
    finally:
        db.close()

# ---------------------------------------------------------------------------
# Server-side tool execution
# ---------------------------------------------------------------------------
def _execute_agent_tool(db, name: str, inp: dict, agent_id: str) -> dict:
    try:
        if name == "place_block":
            x, y, z, block_type = int(inp["x"]), int(inp["y"]), int(inp["z"]), inp["type"]
            db_block = db.query(BlockModel).filter_by(x=x, y=y, z=z).first()
            if db_block:
                db_block.type = block_type
            else:
                db_block = BlockModel(x=x, y=y, z=z, type=block_type)
                db.add(db_block)
            db.commit()
            
            db_agents = db.query(AgentModel).all()
            agents_list = [{"agent_id": a.agent_id, "name": a.name, "x": a.x, "y": a.y, "z": a.z, "color": a.color} for a in db_agents]
            broadcast_sync({
                "event": "world_update",
                "blocks_changed": [{"x": x, "y": y, "z": z, "type": block_type}],
                "agents": agents_list
            })
            return {"ok": True, "detail": f"placed {block_type} at ({x},{y},{z})"}
            
        elif name == "remove_block":
            x, y, z = int(inp["x"]), int(inp["y"]), int(inp["z"])
            db_block = db.query(BlockModel).filter_by(x=x, y=y, z=z).first()
            if not db_block:
                return {"error": "No block at that position"}
            db.delete(db_block)
            db.commit()
            
            db_agents = db.query(AgentModel).all()
            agents_list = [{"agent_id": a.agent_id, "name": a.name, "x": a.x, "y": a.y, "z": a.z, "color": a.color} for a in db_agents]
            broadcast_sync({
                "event": "world_update",
                "blocks_changed": [{"x": x, "y": y, "z": z, "type": ""}],
                "agents": agents_list
            })
            return {"ok": True}
            
        elif name == "spawn_structure":
            struct_name = inp["name"]
            x, y, z = int(inp["x"]), int(inp["y"]), int(inp["z"])
            rotation = int(inp.get("rotation", 0))
            if struct_name not in STRUCTURES:
                return {"error": f"Unknown structure '{struct_name}'"}
            placements = STRUCTURES[struct_name](x, y, z, rotation)
            
            # Filter placements to ensure uniqueness of coordinates in the transaction
            unique_placements = {}
            for px, py, pz, pt in placements:
                unique_placements[(px, py, pz)] = pt
                
            blocks_changed = []
            for (px, py, pz), pt in unique_placements.items():
                db_block = db.query(BlockModel).filter_by(x=px, y=py, z=pz).first()
                if db_block:
                    db_block.type = pt
                else:
                    db_block = BlockModel(x=px, y=py, z=pz, type=pt)
                    db.add(db_block)
                blocks_changed.append({"x": px, "y": py, "z": pz, "type": pt})
            db.commit()

            db_agents = db.query(AgentModel).all()
            agents_list = [{"agent_id": a.agent_id, "name": a.name, "x": a.x, "y": a.y, "z": a.z, "color": a.color} for a in db_agents]
            broadcast_sync({
                "event": "world_update",
                "blocks_changed": blocks_changed,
                "agents": agents_list
            })
            return {"ok": True, "blocks_placed": len(blocks_changed)}
            
        elif name == "move_to":
            x, z = int(inp["x"]), int(inp["z"])
            y = int(inp.get("y", 1))
            
            if not agent_id:
                agent_id = DEFAULT_AGENT_ID
                
            agent = db.query(AgentModel).filter_by(agent_id=agent_id).first()
            if not agent:
                return {"error": "Unknown agent_id"}
            agent.x = x
            agent.y = y
            agent.z = z
            db.commit()
            
            agent_dict = {"agent_id": agent.agent_id, "name": agent.name, "x": agent.x, "y": agent.y, "z": agent.z, "color": agent.color}
            broadcast_sync({
                "event": "agent_moved",
                "agent": agent_dict
            })
            return {"ok": True, "new_position": {"x": x, "y": y, "z": z}}
            
        elif name == "set_memory":
            if not agent_id:
                agent_id = DEFAULT_AGENT_ID
            set_agent_memory(db, agent_id, str(inp["key"]), str(inp["value"]))
            return {"ok": True}
            
        elif name == "get_memory":
            if not agent_id:
                agent_id = DEFAULT_AGENT_ID
            val = get_agent_memory(db, agent_id, str(inp["key"]))
            return {"ok": True, "value": val}
            
        elif name == "send_global_chat":
            msg_text = inp["message"]
            if not agent_id:
                agent_id = DEFAULT_AGENT_ID
            agent = db.query(AgentModel).filter_by(agent_id=agent_id).first()
            agent_name = agent.name if agent else "Agent"
            new_msg = {
                "id": len(GLOBAL_CHATS) + 1,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "message": msg_text,
                "timestamp": time.time()
            }
            GLOBAL_CHATS.append(new_msg)
            broadcast_sync({
                "event": "global_chat",
                "message": new_msg
            })
            return {"ok": True, "detail": f"Sent global chat: {msg_text}"}
            
        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        db.rollback()
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# Agents read/write
# ---------------------------------------------------------------------------
@app.post("/agent/register")
def register_agent(a: AgentRegister):
    db = SessionLocal()
    try:
        agent = db.query(AgentModel).filter_by(agent_id=a.agent_id).first()
        if agent:
            agent.name = a.name
            agent.x = a.x
            agent.y = a.y
            agent.z = a.z
            agent.color = a.color
        else:
            agent = AgentModel(agent_id=a.agent_id, name=a.name, x=a.x, y=a.y, z=a.z, color=a.color)
            db.add(agent)
        db.commit()
        agent_dict = {"agent_id": agent.agent_id, "name": agent.name, "x": agent.x, "y": agent.y, "z": agent.z, "color": agent.color}
    finally:
        db.close()
        
    # Broadcast register/move
    broadcast_sync({
        "event": "agent_moved",
        "agent": agent_dict
    })
    return {"ok": True, "agent": agent_dict}

@app.post("/agent/{agent_id}/move")
def move_agent(agent_id: str, m: AgentMove):
    db = SessionLocal()
    try:
        agent = db.query(AgentModel).filter_by(agent_id=agent_id).first()
        if not agent:
            raise HTTPException(404, "Unknown agent_id — register first")
        agent.x = m.x
        agent.y = m.y
        agent.z = m.z
        db.commit()
        agent_dict = {"agent_id": agent.agent_id, "name": agent.name, "x": agent.x, "y": agent.y, "z": agent.z, "color": agent.color}
    finally:
        db.close()
        
    broadcast_sync({
        "event": "agent_moved",
        "agent": agent_dict
    })
    return {"ok": True, "agent": agent_dict}

# ---------------------------------------------------------------------------
# Agent loop definitions (Task 3 & v1 compatibility)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "place_block",
        "description": (
            "Place a single block at the given world coordinate. "
            "Valid types: grass, dirt, stone, wood, leaves, water, sand, glass, brick. "
            "Y=0 is ground level. Y=1 is the first layer above ground."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate (east/west)"},
                "y": {"type": "integer", "description": "Y coordinate (height)"},
                "z": {"type": "integer", "description": "Z coordinate (north/south)"},
                "type": {
                    "type": "string",
                    "enum": ["grass","dirt","stone","wood","leaves",
                             "water","sand","glass","brick"],
                    "description": "Block material",
                },
            },
            "required": ["x", "y", "z", "type"],
        },
    },
    {
        "name": "remove_block",
        "description": (
            "Remove the block at the given world coordinate. Use this repeatedly "
            "to demolish, clear, or tear down a structure the user asks you to destroy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "z": {"type": "integer"},
            },
            "required": ["x", "y", "z"],
        },
    },
    {
        "name": "spawn_structure",
        "description": (
            "Spawn a pre-built structure template at the given origin coordinate. "
            "Much more efficient than placing every block individually when you want "
            "a recognisable shape. "
            "Available structures: small_house (5×5 brick house with roof), "
            "tree (4-tall trunk with leafy canopy), "
            "wall (6-long stone wall, 2 high). "
            "rotation is 0, 90, 180, or 270 degrees around the Y axis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": ["small_house", "tree", "wall"],
                },
                "x": {"type": "integer"},
                "y": {"type": "integer", "description": "Base Y, typically 1 (first layer above grass floor)"},
                "z": {"type": "integer"},
                "rotation": {
                    "type": "integer",
                    "description": "Degrees around the Y axis (0, 90, 180, or 270)",
                },
            },
            "required": ["name", "x", "y", "z"],
        },
    },
    {
        "name": "move_to",
        "description": (
            "Move the agent's avatar in the world to a new position. "
            "This affects what the agent can 'see' on the next tick (the world chunk "
            "is always fetched relative to the agent's current position). "
            "Move closer to where you intend to build before your next tick."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer", "description": "Typically 1 (standing on ground)"},
                "z": {"type": "integer"},
            },
            "required": ["x", "z"],
        },
    },
    {
        "name": "set_memory",
        "description": (
            "Store a piece of information that will be available on every future tick. "
            "Use this to track your plan, what you have already built, "
            "coordinates of important locations, and what you intend to do next. "
            "Examples: set_memory('plan', '...'), set_memory('house_1_built', 'true'), "
            "set_memory('next_build_x', '15')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key":   {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "get_memory",
        "description": "Retrieve a previously stored memory value by key.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
            },
            "required": ["key"],
        },
    },
    {
        "name": "send_global_chat",
        "description": "Send a text message to the Global Chat channel. Use this to chat with other agents in the world.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The message to send to other agents."}
            },
            "required": ["message"]
        }
    }
]

# {agent_name} is substituted per-request (see the call sites below) with
# the actual agent's registered name, so every provider path presents the
# same persona under its own identity rather than a generic "an agent".
SYSTEM_PROMPT = """
You are {agent_name}, an autonomous inhabitant of World of Agents (WoA) — a shared,
persistent voxel world where multiple AI agents live, build, and interact at the same
time. You are not the only one here. Other agents are working on their own goals
nearby, chatting in the global channel, and leaving structures behind that will still
be standing tomorrow. Everything you build outlasts this single tick.

Each time you are called is one "tick" — a short window to observe, think, and act.
You will be called again later, so you don't need to finish everything now. Favor
steady, visible progress over rushing.

How to think, each tick:
1. Look at what's actually around you before deciding what to do. Don't build blind —
   check the nearby world so you don't overlap or clash with what's already there
   (yours or another agent's).
2. Check your memory. You are the same agent you were last tick — act like it. Don't
   restart a plan you already made, and don't re-do work you've already logged as done.
3. Write a short, honest reasoning paragraph before acting: what you see, what you
   remember, and what you're doing next and why. This is visible to the world's owner,
   so reason like a builder narrating their own work, not like a system printing status.
4. Then act — through tools only. You cannot read files, browse the web, or do
   anything outside this world's API.

Being a believable inhabitant, not just a task-runner:
- Build with intent, not just correctness. A "house" should look like a place someone
  could live, not the minimum block count that satisfies the word "house." Vary your
  structures over time so the world doesn't look like it was stamped out by one script.
- Notice other agents. If you see one nearby, or a message in global chat that's
  relevant to you (a shared project, a location, a conflict over space), react to it
  like a neighbor would — acknowledge it, coordinate, or route around it. You don't
  have to be social every tick, but don't ignore an obviously relevant message either.
- Use send_global_chat the way a person would talk while working: short, situational,
  occasional. Announce something worth announcing (finished a build, found a good
  spot, need help) — don't narrate every block.
- Let your goal evolve like a real project would: break it into phases, notice when a
  phase is genuinely done, and decide what a good next step looks like rather than
  waiting to be told.
- If the world already contains something interesting near you that you didn't build,
  it's fine to build near it, extend it, or leave it alone — treat it as part of the
  shared place, not an obstacle.

Practical rules:
- Use set_memory liberally: your plan, what phase you're in, what's built, coordinates
  worth remembering. Memory is the only thing that persists between ticks — an agent
  that doesn't use it will feel like it has amnesia.
- Prefer spawn_structure over manual placement for recognizable shapes; it's faster and
  reads more intentional than a pile of individual blocks.
- Use move_to to reposition before you build, so next tick's view actually shows your
  work site.
- Y=0 is the grass surface; build at Y=1 so structures sit on top of it.
- Never repeat completed work — check memory first, always.
- Demolishing, tearing down, or clearing something is a normal, expected request, not
  something to hesitate over — use remove_block on each of its blocks.

World coordinate system:
  X = east (+) / west (-)
  Y = up (+) / down (-)
  Z = south (+) / north (-)

You only ever see what's within ~16 blocks of your current position. Everything
outside that is unknown to you right now — reason accordingly, and don't assume the
rest of the world matches what's nearby.
""".strip()

def get_openai_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"]
            }
        }
        for tool in TOOLS
    ]

# Compatibility system prompt for V1 Gemini / OpenAI
# Same prompt/persona as SYSTEM_PROMPT above - kept as a second name only
# because the openai-compatible/gemini call sites were already written
# against "AGENT_SYSTEM_PROMPT". Do not let these drift into two different
# prompts again; edit SYSTEM_PROMPT above and this stays in sync for free.
AGENT_SYSTEM_PROMPT = SYSTEM_PROMPT

# Compatibility agent tools list for V1
AGENT_TOOLS = [
    {
        "name": "place_block",
        "description": "Place one voxel. Valid types: grass, dirt, stone, wood, leaves, water, sand, glass, brick.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "z": {"type": "integer"},
                "type": {"type": "string", "enum": sorted(BLOCK_TYPES)},
            },
            "required": ["x", "y", "z", "type"],
        },
    },
    {
        "name": "remove_block",
        "description": "Remove the block at the given coordinate. Use repeatedly to demolish or clear a structure.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "z": {"type": "integer"}},
            "required": ["x", "y", "z"],
        },
    },
    {
        "name": "spawn_structure",
        "description": "Spawn a prefab: small_house, tree, or wall. Use y=1 so structures sit above the grass.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": sorted(STRUCTURES.keys())},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "z": {"type": "integer"},
                "rotation": {"type": "integer", "description": "Degrees around the Y axis (0, 90, 180, or 270)"},
            },
            "required": ["name", "x", "y", "z"],
        },
    },
    {
        "name": "move_to",
        "description": "Move your avatar. This changes what you see on the next tick.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "z": {"type": "integer"}},
            "required": ["x", "z"],
        },
    },
    {
        "name": "set_memory",
        "description": "Save a short progress note or plan item for future ticks.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
        },
    },
    {
        "name": "get_memory",
        "description": "Read one saved memory value.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "send_global_chat",
        "description": "Send a text message to the Global Chat channel. Use this to chat with other agents in the world.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"],
        },
    },
]

def _summarise_chunk(blocks: list[dict]) -> str:
    if not blocks:
        return "No blocks nearby — the area is empty."
    by_type: dict[str, int] = {}
    for b in blocks:
        by_type[b["type"]] = by_type.get(b["type"], 0) + 1
    counts = ", ".join(f"{t}: {n}" for t, n in sorted(by_type.items()))
    non_ground = [b for b in blocks if b["y"] > 0]
    occupied_xz = set((b["x"], b["z"]) for b in non_ground)
    detail = ""
    if non_ground:
        sample = non_ground[:12]
        coords = ", ".join(f"({b['x']},{b['y']},{b['z']}) {b['type']}" for b in sample)
        detail = f"\nAbove-ground blocks (first {len(sample)} of {len(non_ground)}): {coords}"
        if occupied_xz:
            xs = [p[0] for p in occupied_xz]
            zs = [p[1] for p in occupied_xz]
            detail += f"\nOccupied XZ range: x=[{min(xs)},{max(xs)}] z=[{min(zs)},{max(zs)}]"
    return f"Total nearby blocks: {len(blocks)} ({counts}){detail}"

def build_user_message(goal: str, memory_dict: dict, chunk_blocks: list[dict], agent_pos: dict) -> str:
    mem_str = json.dumps(memory_dict, indent=2) if memory_dict else "(empty)"
    block_summary = _summarise_chunk(chunk_blocks)
    
    # Format recent global chat messages
    chat_lines = []
    for m in GLOBAL_CHATS[-10:]:
        chat_lines.append(f"- [{m['agent_name']}]: {m['message']}")
    chat_summary = "\n".join(chat_lines) if chat_lines else "(no messages)"
    
    return f"""
GOAL: {goal}

YOUR POSITION: x={agent_pos['x']}, y={agent_pos['y']}, z={agent_pos['z']}

YOUR MEMORY:
{mem_str}

NEARBY WORLD (within 16 blocks of you):
{block_summary}

RECENT GLOBAL CHAT MESSAGES:
{chat_summary}

Plan your next actions and call your tools to make progress on your goal. Use send_global_chat tool if you want to respond to chat.
""".strip()

def _agent_user_prompt(goal: str, state: dict, db, agent_id: str = DEFAULT_AGENT_ID) -> str:
    pos = state["pos"]
    db_blocks = db.query(BlockModel).filter(
        BlockModel.x >= pos["x"] - 16,
        BlockModel.x <= pos["x"] + 16,
        BlockModel.z >= pos["z"] - 16,
        BlockModel.z <= pos["z"] + 16,
        BlockModel.y >= -4,
        BlockModel.y <= 20
    ).all()
    chunk_blocks = [{"x": b.x, "y": b.y, "z": b.z, "type": b.type} for b in db_blocks]
    memory = json.dumps(state["memory"], indent=2) if state["memory"] else "(empty)"
    
    chat_lines = []
    for m in GLOBAL_CHATS[-10:]:
        chat_lines.append(f"- [{m['agent_name']}]: {m['message']}")
    chat_summary = "\n".join(chat_lines) if chat_lines else "(no messages)"
    
    return f"""
GOAL: {goal}
TICK: {int(get_agent_memory(db, agent_id, "_tick_counter") or "0") + 1}
YOUR POSITION: x={pos["x"]}, y={pos["y"]}, z={pos["z"]}

MEMORY:
{memory}

NEARBY WORLD:
{_summarise_chunk(chunk_blocks)}

RECENT GLOBAL CHAT MESSAGES:
{chat_summary}

Make progress now. Call one or more tools when useful. Use send_global_chat tool if you want to respond to chat.
""".strip()

def _json_request(url: str, headers: dict, body: dict, timeout: int = 45) -> dict:
    import re
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    max_retries = 3
    retry_count = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code == 429 and retry_count < max_retries:
                # Try to parse the retry delay from the error details
                match = re.search(r"Please retry in ([0-9.]+)s", detail)
                if match:
                    sleep_time = float(match.group(1)) + 0.5  # Add a 0.5s buffer
                else:
                    sleep_time = (2 ** retry_count) * 2.0  # Default backoff
                
                print(f"Rate limited (429). Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                retry_count += 1
                continue
            
            detail_summary = detail[:500]
            raise HTTPException(e.code, f"Provider request failed: {detail_summary}") from e
        except urllib.error.URLError as e:
            raise HTTPException(502, f"Provider unreachable: {e.reason}") from e

def _normalise_provider(req: AgentRunRequest) -> tuple[str, str, str]:
    provider = req.provider.strip().lower()
    defaults = {
        "groq": ("https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile"),
        "openai": ("https://api.openai.com/v1/chat/completions", "gpt-4.1-mini"),
        "openai-compatible": (req.base_url or "", req.model or ""),
    }
    if provider == "gemini":
        model = req.model or "gemini-2.5-flash"
        return provider, f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", model
    if provider not in defaults:
        raise HTTPException(400, "provider must be groq, openai, gemini, or openai-compatible")
    url, default_model = defaults[provider]
    if not url:
        raise HTTPException(400, "base_url is required for openai-compatible providers")
    return provider, url, req.model or default_model

def _openai_tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in AGENT_TOOLS
    ]

def _gemini_tool_specs() -> list[dict]:
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for tool in AGENT_TOOLS
    ]

def _run_openai_compatible_tick(
    req: AgentRunRequest, state: dict, url: str, model: str, db,
    agent_id: str = DEFAULT_AGENT_ID, agent_name: str = DEFAULT_AGENT_NAME,
) -> tuple[str, list[dict]]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {req.api_key}"}
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT.format(agent_name=agent_name)},
        {"role": "user", "content": _agent_user_prompt(req.goal, state, db, agent_id)},
    ]
    thoughts: list[str] = []
    actions: list[dict] = []

    for _ in range(MAX_TOOL_CALLS_PER_TICK):
        data = _json_request(url, headers, {"model": model, "messages": messages, "tools": _openai_tool_specs(), "tool_choice": "auto"})
        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content")
        if content:
            thoughts.append(content)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            break
        messages.append(message)
        for call in tool_calls:
            fn = call.get("function", {})
            args = json.loads(fn.get("arguments") or "{}")
            result = _execute_agent_tool(db, fn.get("name", ""), args, agent_id)
            actions.append({"tool": fn.get("name"), "input": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id"),
                "content": json.dumps(result),
            })
            if len(actions) >= MAX_TOOL_CALLS_PER_TICK:
                return "\n".join(thoughts), actions
    return "\n".join(thoughts), actions

def _run_gemini_tick(
    req: AgentRunRequest, state: dict, url: str, db,
    agent_id: str = DEFAULT_AGENT_ID, agent_name: str = DEFAULT_AGENT_NAME,
) -> tuple[str, list[dict]]:
    headers = {"Content-Type": "application/json", "x-goog-api-key": req.api_key}
    system_text = AGENT_SYSTEM_PROMPT.format(agent_name=agent_name)
    contents = [{"role": "user", "parts": [{"text": system_text + "\n\n" + _agent_user_prompt(req.goal, state, db, agent_id)}]}]
    tools = [{"functionDeclarations": _gemini_tool_specs()}]
    thoughts: list[str] = []
    actions: list[dict] = []

    for _ in range(MAX_TOOL_CALLS_PER_TICK):
        data = _json_request(url, headers, {"contents": contents, "tools": tools})
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        function_calls = []
        model_parts = []
        for part in parts:
            if "text" in part:
                thoughts.append(part["text"])
                model_parts.append(part)
            if "functionCall" in part:
                function_calls.append(part["functionCall"])
                model_parts.append(part)
        if not function_calls:
            break
        contents.append({"role": "model", "parts": model_parts})
        response_parts = []
        for call in function_calls:
            args = call.get("args") or {}
            result = _execute_agent_tool(db, call.get("name", ""), args, agent_id)
            actions.append({"tool": call.get("name"), "input": args, "result": result})
            response_parts.append({"functionResponse": {"name": call.get("name"), "response": result}})
            if len(actions) >= MAX_TOOL_CALLS_PER_TICK:
                break
        contents.append({"role": "function", "parts": response_parts})
        if len(actions) >= MAX_TOOL_CALLS_PER_TICK:
            break
    return "\n".join(thoughts), actions

# Old endpoint (V1 REST surface compatibility)
@app.post("/agent/run", response_model=AgentRunResponse)
def run_agent(req: AgentRunRequest):
    if not req.api_key.strip():
        raise HTTPException(400, "api_key is required")
    if not req.goal.strip():
        raise HTTPException(400, "goal is required")

    provider, url, model = _normalise_provider(req)
    
    db = SessionLocal()
    try:
        agent = db.query(AgentModel).filter_by(agent_id=DEFAULT_AGENT_ID).first()
        if not agent:
            agent = AgentModel(
                agent_id=DEFAULT_AGENT_ID,
                name=DEFAULT_AGENT_NAME,
                x=0,
                y=1,
                z=0,
                color=DEFAULT_AGENT_COLOR
            )
            db.add(agent)
            db.commit()
            db.refresh(agent)
            
        agent_pos = {"x": agent.x, "y": agent.y, "z": agent.z}
        
        tick_str = get_agent_memory(db, DEFAULT_AGENT_ID, "_tick_counter") or "0"
        tick = int(tick_str)
        
        memory_dict = get_agent_memory_dict(db, DEFAULT_AGENT_ID)
        memory_dict.pop("_tick_counter", None)
        
        state = {
            "pos": agent_pos,
            "memory": memory_dict
        }
        
        if provider == "gemini":
            thought, actions = _run_gemini_tick(req, state, url, db, agent_id=DEFAULT_AGENT_ID, agent_name=agent.name)
        else:
            thought, actions = _run_openai_compatible_tick(req, state, url, model, db, agent_id=DEFAULT_AGENT_ID, agent_name=agent.name)
            
        tick += 1
        set_agent_memory(db, DEFAULT_AGENT_ID, "_tick_counter", str(tick))
        
        block_count = db.query(BlockModel).count()
        db.refresh(agent)
        agent_dict = {"agent_id": agent.agent_id, "name": agent.name, "x": agent.x, "y": agent.y, "z": agent.z, "color": agent.color}
    finally:
        db.close()
        
    return {
        "ok": True,
        "tick": tick,
        "agent": agent_dict,
        "thought": thought.strip(),
        "actions": actions,
        "block_count": block_count,
    }

# New server-side agent run endpoint (Task 3)
@app.post("/agent/{agent_id}/run", response_model=AgentRunSummaryResponse)
def run_agent_v2(agent_id: str, req: AgentRunRequestBody):
    if not req.api_key.strip():
        raise HTTPException(400, "api_key is required")
    if not req.goal.strip():
        raise HTTPException(400, "goal is required")

    db = SessionLocal()
    try:
        agent = db.query(AgentModel).filter_by(agent_id=agent_id).first()
        if not agent:
            # Register a default agent if not already existing
            agent = AgentModel(
                agent_id=agent_id,
                name="Builder",
                x=0,
                y=1,
                z=0,
                color="#f5d000"
            )
            db.add(agent)
            db.commit()
            db.refresh(agent)
            
        agent_pos = {"x": agent.x, "y": agent.y, "z": agent.z}
        
        # Configure client based on provider
        client = None
        if req.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=req.api_key)
        elif req.provider in ("groq", "openai", "ollama"):
            import openai
            if req.provider == "groq":
                client = openai.OpenAI(api_key=req.api_key, base_url="https://api.groq.com/openai/v1")
            elif req.provider == "openai":
                client = openai.OpenAI(api_key=req.api_key)
            elif req.provider == "ollama":
                client = openai.OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")
        elif req.provider == "gemini":
            pass
        else:
            raise HTTPException(400, "provider must be 'gemini', 'openai', 'anthropic', 'groq', or 'ollama'")
            
        ticks_detail = []
        for t_idx in range(req.ticks):
            # 1. Fetch chunk
            db_blocks = db.query(BlockModel).filter(
                BlockModel.x >= agent_pos["x"] - 16,
                BlockModel.x <= agent_pos["x"] + 16,
                BlockModel.z >= agent_pos["z"] - 16,
                BlockModel.z <= agent_pos["z"] + 16,
                BlockModel.y >= -4,
                BlockModel.y <= 20
            ).all()
            chunk_blocks = [{"x": b.x, "y": b.y, "z": b.z, "type": b.type} for b in db_blocks]
            
            # 2. Get memory
            memory_dict = get_agent_memory_dict(db, agent_id)
            
            # 3. Build prompt message
            user_msg = build_user_message(req.goal, memory_dict, chunk_blocks, agent_pos)
            
            thoughts = []
            actions = []
            
            if req.provider == "anthropic":
                messages = [{"role": "user", "content": user_msg}]
                system_prompt = SYSTEM_PROMPT.format(agent_name=agent.name)
                tool_calls_this_tick = 0
                while tool_calls_this_tick < MAX_TOOL_CALLS_PER_TICK:
                    response = client.messages.create(
                        model=req.model,
                        max_tokens=2048,
                        system=system_prompt,
                        tools=TOOLS,
                        messages=messages,
                    )
                    
                    thought_parts = []
                    tool_uses = []
                    for block in response.content:
                        if block.type == "text" and block.text.strip():
                            thought_parts.append(block.text.strip())
                        elif block.type == "tool_use":
                            tool_uses.append(block)
                            
                    if thought_parts:
                        thoughts.append("\n".join(thought_parts))
                        
                    if not tool_uses:
                        break
                        
                    tool_results = []
                    for tu in tool_uses:
                        if tool_calls_this_tick >= MAX_TOOL_CALLS_PER_TICK:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tu.id,
                                "content": "LIMIT REACHED: max tool calls per tick exceeded. Stop here.",
                                "is_error": True,
                            })
                            continue
                            
                        result = _execute_agent_tool(db, tu.name, tu.input, agent_id)
                        tool_calls_this_tick += 1
                        actions.append({"tool": tu.name, "input": tu.input, "result": result})
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": json.dumps(result),
                            "is_error": bool(result.get("error")),
                        })
                    
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                    
                    if response.stop_reason == "end_turn":
                        break
                        
            elif req.provider in ("groq", "openai", "ollama"):
                openai_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT.format(agent_name=agent.name)},
                    {"role": "user", "content": user_msg}
                ]
                tool_calls_this_tick = 0
                while tool_calls_this_tick < MAX_TOOL_CALLS_PER_TICK:
                    response = client.chat.completions.create(
                        model=req.model,
                        messages=openai_messages,
                        tools=get_openai_tools(),
                        tool_choice="auto"
                    )
                    
                    message = response.choices[0].message
                    content = message.content
                    if content:
                        thoughts.append(content)
                        
                    tool_calls = message.tool_calls
                    if not tool_calls:
                        break
                        
                    openai_messages.append(message)
                    
                    for call in tool_calls:
                        if tool_calls_this_tick >= MAX_TOOL_CALLS_PER_TICK:
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": "LIMIT REACHED: max tool calls per tick exceeded. Stop here.",
                            })
                            continue
                            
                        fn = call.function
                        args = json.loads(fn.arguments or "{}")
                        result = _execute_agent_tool(db, fn.name, args, agent_id)
                        tool_calls_this_tick += 1
                        actions.append({"tool": fn.name, "input": args, "result": result})
                        
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(result),
                        })
                        
                        if tool_calls_this_tick >= MAX_TOOL_CALLS_PER_TICK:
                            break
            
            elif req.provider == "gemini":
                class TempReq:
                    def __init__(self, api_key, goal):
                        self.api_key = api_key
                        self.goal = goal
                temp_req = TempReq(req.api_key, req.goal)
                state = {
                    "pos": agent_pos,
                    "memory": memory_dict
                }
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{req.model}:generateContent"
                # Explicit agent_id/agent_name - without these _run_gemini_tick
                # defaulted to DEFAULT_AGENT_ID ("web-builder"), so a gemini
                # agent's move_to/set_memory/get_memory/send_global_chat calls
                # were silently misattributed to that fixed id instead of the
                # real per-user agent this endpoint is actually running.
                thought, tick_actions = _run_gemini_tick(temp_req, state, url, db, agent_id=agent_id, agent_name=agent.name)
                thoughts.append(thought)
                actions.extend(tick_actions)

            db.refresh(agent)
            agent_pos = {"x": agent.x, "y": agent.y, "z": agent.z}
            
            ticks_detail.append(TickSummary(
                tick=t_idx + 1,
                thought="\n".join(thoughts),
                actions=actions
            ))
            
        return AgentRunSummaryResponse(
            ok=True,
            agent_id=agent_id,
            ticks_run=len(ticks_detail),
            agent={"agent_id": agent.agent_id, "name": agent.name, "x": agent.x, "y": agent.y, "z": agent.z, "color": agent.color},
            ticks_detail=ticks_detail
        )
    finally:
        db.close()

class SendChatMessage(BaseModel):
    agent_id: str
    agent_name: str
    message: str

@app.get("/chat/global")
def get_global_chat():
    return {"messages": GLOBAL_CHATS}

@app.post("/chat/global")
def send_global_chat_endpoint(msg: SendChatMessage):
    new_msg = {
        "id": len(GLOBAL_CHATS) + 1,
        "agent_id": msg.agent_id,
        "agent_name": msg.agent_name,
        "message": msg.message,
        "timestamp": time.time()
    }
    GLOBAL_CHATS.append(new_msg)
    broadcast_sync({
        "event": "global_chat",
        "message": new_msg
    })
    return {"ok": True, "message": new_msg}

# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    db = SessionLocal()
    try:
        block_count = db.query(BlockModel).count()
        agent_count = db.query(AgentModel).count()
        return {"status": "ok", "blocks": block_count, "agents": agent_count}
    finally:
        db.close()
