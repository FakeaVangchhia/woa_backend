"""
Voxel World backend (v1 prototype)

In-memory voxel world. Exposes a small REST API that both a human (via the
Three.js frontend) and, later, an LLM agent (via tool calls) can use to read
and modify the world. The voxel grid is the shared "ground truth" — the
frontend is just a renderer of whatever this API says exists.
"""

import json
import urllib.error
import urllib.request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
agent_run_state: dict[str, dict] = {}

WORLD_STARTED_AT = time.time()
DEFAULT_AGENT_ID = "web-builder"
DEFAULT_AGENT_NAME = "Builder"
DEFAULT_AGENT_COLOR = "#f5d000"
MAX_TOOL_CALLS_PER_TICK = 8


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


# ---------------------------------------------------------------------------
# Web agent runner — one HTTP request = one agent tick.
# ---------------------------------------------------------------------------

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
        "description": "Remove the block at the given coordinate.",
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
                "rotation": {"type": "integer", "enum": [0, 90, 180, 270]},
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
]

AGENT_SYSTEM_PROMPT = """
You are an autonomous builder agent living inside a shared 3D voxel world.
Each request is one tick. Think briefly, then use tools to make visible progress.
Prefer spawn_structure for houses, trees, and walls. Use memory to avoid repeating
work. Coordinates use X east/west, Y up/down, and Z north/south. Y=0 is grass;
place structures at Y=1.
""".strip()


def _summarise_chunk(blocks: list[dict]) -> str:
    if not blocks:
        return "No blocks nearby."
    by_type: dict[str, int] = {}
    above_ground = []
    for b in blocks:
        by_type[b["type"]] = by_type.get(b["type"], 0) + 1
        if b["y"] > 0:
            above_ground.append(b)
    counts = ", ".join(f"{t}: {n}" for t, n in sorted(by_type.items()))
    if not above_ground:
        return f"Total nearby blocks: {len(blocks)} ({counts}). No above-ground structures."
    sample = ", ".join(f"({b['x']},{b['y']},{b['z']}) {b['type']}" for b in above_ground[:16])
    return f"Total nearby blocks: {len(blocks)} ({counts}). Above-ground sample: {sample}"


def _agent_state() -> dict:
    state = agent_run_state.setdefault(
        DEFAULT_AGENT_ID,
        {"tick": 0, "pos": {"x": 0, "y": 1, "z": 0}, "memory": {}},
    )
    if DEFAULT_AGENT_ID not in agents:
        pos = state["pos"]
        agents[DEFAULT_AGENT_ID] = {
            "agent_id": DEFAULT_AGENT_ID,
            "name": DEFAULT_AGENT_NAME,
            "x": pos["x"],
            "y": pos["y"],
            "z": pos["z"],
            "color": DEFAULT_AGENT_COLOR,
        }
    return state


def _agent_user_prompt(goal: str, state: dict) -> str:
    pos = state["pos"]
    chunk = get_chunk(pos["x"], pos["z"])
    memory = json.dumps(state["memory"], indent=2) if state["memory"] else "(empty)"
    return f"""
GOAL: {goal}
TICK: {state["tick"] + 1}
YOUR POSITION: x={pos["x"]}, y={pos["y"]}, z={pos["z"]}

MEMORY:
{memory}

NEARBY WORLD:
{_summarise_chunk(chunk["blocks"])}

Make progress now. Call one or more tools when useful.
""".strip()


def _json_request(url: str, headers: dict, body: dict, timeout: int = 45) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise HTTPException(e.code, f"Provider request failed: {detail}") from e
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


def _run_openai_compatible_tick(req: AgentRunRequest, state: dict, url: str, model: str) -> tuple[str, list[dict]]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {req.api_key}"}
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": _agent_user_prompt(req.goal, state)},
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
            result = _execute_agent_tool(fn.get("name", ""), args, state)
            actions.append({"tool": fn.get("name"), "input": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id"),
                "content": json.dumps(result),
            })
            if len(actions) >= MAX_TOOL_CALLS_PER_TICK:
                return "\n".join(thoughts), actions
    return "\n".join(thoughts), actions


def _run_gemini_tick(req: AgentRunRequest, state: dict, url: str) -> tuple[str, list[dict]]:
    headers = {"Content-Type": "application/json", "x-goog-api-key": req.api_key}
    contents = [{"role": "user", "parts": [{"text": AGENT_SYSTEM_PROMPT + "\n\n" + _agent_user_prompt(req.goal, state)}]}]
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
                model_parts.append({"text": part["text"]})
            if "functionCall" in part:
                function_calls.append(part["functionCall"])
                model_parts.append({"functionCall": part["functionCall"]})
        if not function_calls:
            break
        contents.append({"role": "model", "parts": model_parts})
        response_parts = []
        for call in function_calls:
            args = call.get("args") or {}
            result = _execute_agent_tool(call.get("name", ""), args, state)
            actions.append({"tool": call.get("name"), "input": args, "result": result})
            response_parts.append({"functionResponse": {"name": call.get("name"), "response": result}})
            if len(actions) >= MAX_TOOL_CALLS_PER_TICK:
                break
        contents.append({"role": "function", "parts": response_parts})
        if len(actions) >= MAX_TOOL_CALLS_PER_TICK:
            break
    return "\n".join(thoughts), actions


def _execute_agent_tool(name: str, inp: dict, state: dict) -> dict:
    try:
        if name == "place_block":
            block = Block(x=inp["x"], y=inp["y"], z=inp["z"], type=inp["type"])
            place_block(block)
            return {"ok": True, "detail": f"placed {block.type} at ({block.x},{block.y},{block.z})"}
        if name == "remove_block":
            remove_block(BlockDelete(x=inp["x"], y=inp["y"], z=inp["z"]))
            return {"ok": True}
        if name == "spawn_structure":
            spawn = StructureSpawn(
                name=inp["name"],
                x=inp["x"],
                y=inp["y"],
                z=inp["z"],
                rotation=inp.get("rotation", 0),
            )
            result = spawn_structure(spawn)
            return {"ok": True, "blocks_placed": result["blocks_placed"]}
        if name == "move_to":
            pos = {"x": int(inp["x"]), "y": int(inp.get("y", 1)), "z": int(inp["z"])}
            state["pos"] = pos
            agents[DEFAULT_AGENT_ID].update(pos)
            return {"ok": True, "new_position": pos}
        if name == "set_memory":
            state["memory"][str(inp["key"])] = str(inp["value"])
            return {"ok": True}
        if name == "get_memory":
            return {"ok": True, "value": state["memory"].get(str(inp["key"]))}
        return {"error": f"Unknown tool: {name}"}
    except HTTPException as e:
        return {"error": e.detail}
    except Exception as e:
        return {"error": str(e)}


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


@app.post("/agent/run", response_model=AgentRunResponse)
def run_agent(req: AgentRunRequest):
    if not req.api_key.strip():
        raise HTTPException(400, "api_key is required")
    if not req.goal.strip():
        raise HTTPException(400, "goal is required")

    provider, url, model = _normalise_provider(req)
    state = _agent_state()

    if provider == "gemini":
        thought, actions = _run_gemini_tick(req, state, url)
    else:
        thought, actions = _run_openai_compatible_tick(req, state, url, model)

    state["tick"] += 1
    return {
        "ok": True,
        "tick": state["tick"],
        "agent": agents[DEFAULT_AGENT_ID],
        "thought": thought.strip(),
        "actions": actions,
        "block_count": len(world),
    }


@app.get("/health")
def health():
    return {"status": "ok", "blocks": len(world), "agents": len(agents)}
