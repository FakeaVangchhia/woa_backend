"""
Voxel World backend (v1 prototype)

In-memory voxel world. Exposes a small REST API that both a human (via the
Three.js frontend) and, later, an LLM agent (via tool calls) can use to read
and modify the world. The voxel grid is the shared "ground truth" — the
frontend is just a renderer of whatever this API says exists.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time

app = FastAPI(title="Voxel World API", version="0.1.0")

# Wide-open CORS for local prototyping. Tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# World state
# ---------------------------------------------------------------------------
# Sparse voxel grid: keys are "x,y,z" integer-coordinate strings, values are
# block type strings. Sparse + dict-based is fine for a prototype; swap for
# a chunked store (e.g. one dict per 16x16x16 region) once the world gets big.

BLOCK_TYPES = {"grass", "dirt", "stone", "wood", "leaves", "water", "sand", "glass", "brick"}

world: dict[str, str] = {}

agents: dict[str, dict] = {}  # agent_id -> {x, y, z, name, color}

WORLD_STARTED_AT = time.time()


def _key(x: int, y: int, z: int) -> str:
    return f"{x},{y},{z}"


def _seed_world():
    """Lay down a simple flat grass island so there's something to stand on."""
    size = 24
    for x in range(-size, size + 1):
        for z in range(-size, size + 1):
            world[_key(x, 0, z)] = "grass"
            world[_key(x, -1, z)] = "dirt"


_seed_world()


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


# ---------------------------------------------------------------------------
# World read/write endpoints
# ---------------------------------------------------------------------------

@app.get("/world")
def get_world():
    """Full world snapshot. Fine at prototype scale; chunk this later."""
    return {
        "blocks": [
            {"x": int(k.split(",")[0]), "y": int(k.split(",")[1]), "z": int(k.split(",")[2]), "type": v}
            for k, v in world.items()
        ],
        "agents": list(agents.values()),
        "block_count": len(world),
        "uptime_seconds": round(time.time() - WORLD_STARTED_AT, 1),
    }


@app.get("/world/chunk")
def get_chunk(x: int, z: int, radius: int = 16, y_min: int = -4, y_max: int = 20):
    """Localized view, e.g. what an agent standing at (x, *, z) can 'see'."""
    blocks = []
    for k, v in world.items():
        bx, by, bz = (int(p) for p in k.split(","))
        if abs(bx - x) <= radius and abs(bz - z) <= radius and y_min <= by <= y_max:
            blocks.append({"x": bx, "y": by, "z": bz, "type": v})
    return {"blocks": blocks}


@app.post("/block")
def place_block(block: Block):
    if block.type not in BLOCK_TYPES:
        raise HTTPException(400, f"Unknown block type '{block.type}'. Valid types: {sorted(BLOCK_TYPES)}")
    world[_key(block.x, block.y, block.z)] = block.type
    return {"ok": True, "block": block}


@app.delete("/block")
def remove_block(block: BlockDelete):
    k = _key(block.x, block.y, block.z)
    if k not in world:
        raise HTTPException(404, "No block at that position")
    del world[k]
    return {"ok": True}


# ---------------------------------------------------------------------------
# Structures — prefab templates an agent can spawn in one call instead of
# placing every block individually.
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
    w, d, h = 5, 5, 3  # interior-ish footprint
    # floor
    for x in range(w):
        for z in range(d):
            blocks.append((x, 0, z, "wood"))
    # walls (brick), leave a door gap at z=0, x=2
    for y in range(1, h):
        for x in range(w):
            for z in (0, d - 1):
                if y == 1 and x == 2 and z == 0:
                    continue  # doorway
                blocks.append((x, y, z, "brick"))
        for z in range(d):
            for x in (0, w - 1):
                blocks.append((x, y, x and z or z, "brick"))
    # simpler: redo side walls correctly
    blocks = [b for b in blocks if not (b[0] in (0, w - 1) and b[2] not in (0, d - 1) and b[1] >= 1)]
    for y in range(1, h):
        for z in range(d):
            for x in (0, w - 1):
                blocks.append((x, y, z, "brick"))
    # windows: punch glass into the long walls at y=1
    blocks = [b for b in blocks if not (b[1] == 1 and b[0] in (0, w - 1) and b[2] == 2)]
    for z in (2,):
        for x in (0, w - 1):
            blocks.append((x, 1, z, "glass"))
    # roof
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
    placements = STRUCTURES[s.name](s.x, s.y, s.z, s.rotation)
    for x, y, z, t in placements:
        world[_key(x, y, z)] = t
    return {"ok": True, "blocks_placed": len(placements)}


# ---------------------------------------------------------------------------
# Agents (avatars in the world — minimal for now, just position + identity)
# ---------------------------------------------------------------------------

@app.post("/agent/register")
def register_agent(a: AgentRegister):
    agents[a.agent_id] = {"agent_id": a.agent_id, "name": a.name, "x": a.x, "y": a.y, "z": a.z, "color": a.color}
    return {"ok": True, "agent": agents[a.agent_id]}


@app.post("/agent/{agent_id}/move")
def move_agent(agent_id: str, m: AgentMove):
    if agent_id not in agents:
        raise HTTPException(404, "Unknown agent_id — register first")
    agents[agent_id].update({"x": m.x, "y": m.y, "z": m.z})
    return {"ok": True, "agent": agents[agent_id]}


@app.get("/health")
def health():
    return {"status": "ok", "blocks": len(world), "agents": len(agents)}
