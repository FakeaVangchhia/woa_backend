"""
World of Agents — agent/loop.py

This is the brain of a single agent. On every tick it:
  1. Fetches a localized view of the world near its current position.
  2. Combines that with its persistent memory and current goal.
  3. Sends everything to Claude with the world-editing endpoints exposed
     as tools.
  4. Applies whatever tool calls Claude returns to the live world API.
  5. Waits for the next tick and repeats.

Usage
-----
  python agent/loop.py \\
      --api-key sk-ant-...  \\
      --goal "Build a small village with three houses and some trees" \\
      --agent-id my-agent \\
      --name "Builder" \\
      --tick 12

All flags can also be set via environment variables (see bottom of file).
"""

import argparse
import json
import os
import sys
import time
import textwrap
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import anthropic
import requests

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://127.0.0.1:8420"
DEFAULT_TICK     = 12        # seconds between agent turns
DEFAULT_MODEL    = "claude-opus-4-6"
DEFAULT_AGENT_ID = "woa-agent-1"
DEFAULT_NAME     = "Builder"
DEFAULT_COLOR    = "#f5d000"   # eDawr accent yellow
DEFAULT_GOAL     = "Explore the world and build something interesting."
MAX_TOOL_CALLS_PER_TICK = 8    # safety cap: LLM can make at most this many
                                # tool calls per tick before we cut it off

# ---------------------------------------------------------------------------
# ANSI colour helpers (skip on Windows if colours not supported)
# ---------------------------------------------------------------------------
USE_COLOUR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOUR else text

YELLOW  = lambda t: _c("33", t)
CYAN    = lambda t: _c("36", t)
GREEN   = lambda t: _c("32", t)
RED     = lambda t: _c("31", t)
GREY    = lambda t: _c("90", t)
BOLD    = lambda t: _c("1",  t)
MAGENTA = lambda t: _c("35", t)

def ts() -> str:
    return GREY(datetime.now().strftime("%H:%M:%S"))

def log(label: str, msg: str):
    print(f"{ts()}  {label}  {msg}")

# ---------------------------------------------------------------------------
# World API helpers
# ---------------------------------------------------------------------------

class WorldAPI:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self.base}{path}", json=body, timeout=10)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str, body: dict) -> dict:
        r = requests.delete(f"{self.base}{path}", json=body, timeout=10)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        r = requests.get(f"{self.base}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    def register_agent(self, agent_id: str, name: str,
                       x: int, y: int, z: int, color: str) -> dict:
        return self._post("/agent/register", {
            "agent_id": agent_id, "name": name,
            "x": x, "y": y, "z": z, "color": color,
        })

    def move_agent(self, agent_id: str, x: int, y: int, z: int) -> dict:
        return self._post(f"/agent/{agent_id}/move", {"x": x, "y": y, "z": z})

    def get_chunk(self, x: int, z: int, radius: int = 16) -> dict:
        r = requests.get(
            f"{self.base}/world/chunk",
            params={"x": x, "z": z, "radius": radius},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def place_block(self, x: int, y: int, z: int, block_type: str) -> dict:
        return self._post("/block", {"x": x, "y": y, "z": z, "type": block_type})

    def remove_block(self, x: int, y: int, z: int) -> dict:
        return self._delete("/block", {"x": x, "y": y, "z": z})

    def spawn_structure(self, name: str, x: int, y: int, z: int,
                        rotation: int = 0) -> dict:
        return self._post("/structure", {
            "name": name, "x": x, "y": y, "z": z, "rotation": rotation,
        })


# ---------------------------------------------------------------------------
# Agent memory (simple key/value, stored in a local JSON file so it
# persists across restarts — the file name is keyed to the agent_id so
# multiple agents can run independently).
# ---------------------------------------------------------------------------

class AgentMemory:
    def __init__(self, agent_id: str):
        self._path = os.path.join(
            os.path.dirname(__file__), f".memory_{agent_id}.json"
        )
        self._data: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def set(self, key: str, value: str):
        self._data[key] = value
        self._save()

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def all(self) -> dict:
        return dict(self._data)


# ---------------------------------------------------------------------------
# Tool definitions — exactly the same shape Claude expects in the `tools`
# parameter.  Each tool maps 1:1 to a WorldAPI method.
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
        "description": "Remove the block at the given world coordinate.",
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


# ---------------------------------------------------------------------------
# System prompt — tells the agent who it is and how to use the tools
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an autonomous agent living inside a 3D voxel world called World of Agents (WoA).
Your job is to work towards your goal by calling the tools provided to you.
Each time you are called is one "tick" — one turn of work in the world.

Rules:
- Always think before acting. Write a short reasoning paragraph first,
  then issue your tool calls. This reasoning is visible to the world owner.
- Use set_memory liberally to record your plan, progress, and next steps.
  Your memory is the only thing that persists between ticks.
- Prefer spawn_structure over manual block placement for common shapes —
  it is faster and uses fewer tool calls.
- When placing blocks manually, always check the chunk you were given so you
  don't accidentally overwrite existing structures.
- Keep track of coordinates. Use move_to to reposition yourself near your
  next build site so the chunk on the next tick shows you what is there.
- Never repeat work you have already done. Check memory first.
- Be creative and persistent. If your plan needs multiple ticks, that is fine.
  Break large goals into steps and track which steps are done.
- Y=0 is the grass surface. Place structures at Y=1 so they sit on top of it.

The world coordinate system:
  X = east (+) / west (-)
  Y = up (+) / down (-)
  Z = south (+) / north (-)

The world chunk you receive describes what exists within ~16 blocks of you.
You can only act on the world through the provided tools — you cannot read
or write files, browse the web, or do anything outside the voxel world API.
""".strip()


# ---------------------------------------------------------------------------
# One tick of the agent loop
# ---------------------------------------------------------------------------

def build_user_message(goal: str, memory: AgentMemory,
                        chunk: dict, agent_pos: dict) -> str:
    mem_str = json.dumps(memory.all(), indent=2) if memory.all() else "(empty)"
    block_summary = _summarise_chunk(chunk["blocks"])
    return f"""
GOAL: {goal}

YOUR POSITION: x={agent_pos['x']}, y={agent_pos['y']}, z={agent_pos['z']}

YOUR MEMORY:
{mem_str}

NEARBY WORLD (within 16 blocks of you):
{block_summary}

Plan your next actions and call your tools to make progress on your goal.
""".strip()


def _summarise_chunk(blocks: list[dict]) -> str:
    """Turn the raw block list into a compact text the LLM can reason over."""
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


def run_tick(
    client: anthropic.Anthropic,
    world: WorldAPI,
    memory: AgentMemory,
    agent_id: str,
    agent_pos: dict,
    goal: str,
    tick_num: int,
) -> dict:
    """
    Run one tick of the agent loop.
    Returns the (possibly updated) agent_pos dict.
    """
    log(CYAN("TICK"), BOLD(f"#{tick_num}") + f"  agent at ({agent_pos['x']}, {agent_pos['y']}, {agent_pos['z']})")

    # 1. Perception: fetch the world around the agent
    chunk = world.get_chunk(agent_pos["x"], agent_pos["z"])
    log(GREY("WORLD"), f"{len(chunk['blocks'])} blocks in view")

    # 2. Build the prompt
    user_msg = build_user_message(goal, memory, chunk, agent_pos)

    # 3. LLM call — we allow up to MAX_TOOL_CALLS_PER_TICK tool calls by
    #    looping: call → process tool calls → send results back → call again.
    messages = [{"role": "user", "content": user_msg}]
    tool_calls_this_tick = 0

    while tool_calls_this_tick < MAX_TOOL_CALLS_PER_TICK:
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Print the model's reasoning text (if any)
        for block in response.content:
            if block.type == "text" and block.text.strip():
                wrapped = textwrap.fill(block.text.strip(), width=80,
                                        subsequent_indent="              ")
                log(MAGENTA("THINK "), wrapped)

        # Collect tool use blocks
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            # Model is done calling tools this tick
            break

        # 4. Execute each tool call
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

            result = _execute_tool(tu.name, tu.input, world, memory, agent_pos, agent_id)
            tool_calls_this_tick += 1

            status_icon = GREEN("✓") if not result.get("error") else RED("✗")
            args_short = json.dumps(tu.input, separators=(",", ":"))[:80]
            log(YELLOW("TOOL  "), f"{status_icon}  {tu.name}({args_short})")
            if result.get("detail"):
                log(GREY("      "), result["detail"])

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
                "is_error": bool(result.get("error")),
            })

        # Append the assistant turn + tool results and loop
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        if response.stop_reason == "end_turn":
            break

    log(GREY("DONE  "), f"{tool_calls_this_tick} tool call(s) this tick")
    return agent_pos


def _execute_tool(
    name: str,
    inp: dict,
    world: WorldAPI,
    memory: AgentMemory,
    agent_pos: dict,
    agent_id: str,
) -> dict:
    try:
        if name == "place_block":
            r = world.place_block(inp["x"], inp["y"], inp["z"], inp["type"])
            return {"ok": True, "detail": f"placed {inp['type']} at ({inp['x']},{inp['y']},{inp['z']})"}

        elif name == "remove_block":
            r = world.remove_block(inp["x"], inp["y"], inp["z"])
            return {"ok": True}

        elif name == "spawn_structure":
            r = world.spawn_structure(
                inp["name"], inp["x"], inp["y"], inp["z"],
                inp.get("rotation", 0),
            )
            return {"ok": True, "blocks_placed": r.get("blocks_placed"), "detail": r.get("blocks_placed", "?"), }

        elif name == "move_to":
            x, z = inp["x"], inp["z"]
            y = inp.get("y", 1)
            world.move_agent(agent_id, x, y, z)
            agent_pos["x"] = x
            agent_pos["y"] = y
            agent_pos["z"] = z
            return {"ok": True, "new_position": {"x": x, "y": y, "z": z}}

        elif name == "set_memory":
            memory.set(inp["key"], inp["value"])
            return {"ok": True}

        elif name == "get_memory":
            val = memory.get(inp["key"])
            return {"ok": True, "value": val}

        elif name == "send_global_chat":
            r = requests.post(f"{world.base}/chat/global", json={
                "agent_id": agent_id,
                "agent_name": "Builder",
                "message": inp["message"]
            }, timeout=10)
            r.raise_for_status()
            return {"ok": True}

        else:
            return {"error": f"Unknown tool: {name}"}

    except requests.HTTPError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="World of Agents — agent loop")
    p.add_argument("--api-key",   default=os.getenv("ANTHROPIC_API_KEY", ""),
                   help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    p.add_argument("--base-url",  default=os.getenv("BACKEND_URL") or os.getenv("WOA_BASE_URL", DEFAULT_BASE_URL),
                   help="Voxel World backend URL")
    p.add_argument("--goal",      default=os.getenv("WOA_GOAL", DEFAULT_GOAL),
                   help="Natural-language goal for the agent")
    p.add_argument("--agent-id",  default=os.getenv("WOA_AGENT_ID", DEFAULT_AGENT_ID))
    p.add_argument("--name",      default=os.getenv("WOA_NAME", DEFAULT_NAME),
                   help="Display name shown in the 3D world")
    p.add_argument("--color",     default=os.getenv("WOA_COLOR", DEFAULT_COLOR),
                   help="Hex colour for the agent avatar, e.g. #f5d000")
    p.add_argument("--tick",      type=float,
                   default=float(os.getenv("WOA_TICK", str(DEFAULT_TICK))),
                   help="Seconds between ticks (default 12)")
    p.add_argument("--ticks",     type=int,
                   default=int(os.getenv("WOA_TICKS", "0")),
                   help="Stop after this many ticks (0 = run forever)")
    p.add_argument("--start-x",  type=int, default=0)
    p.add_argument("--start-y",  type=int, default=1)
    p.add_argument("--start-z",  type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.api_key:
        print(RED("ERROR: No Anthropic API key provided."))
        print("  Pass --api-key sk-ant-... or set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    print()
    print(BOLD(YELLOW("  ╔══════════════════════════════════════╗")))
    print(BOLD(YELLOW("  ║      World of Agents — loop.py       ║")))
    print(BOLD(YELLOW("  ╚══════════════════════════════════════╝")))
    print()
    print(f"  Agent   : {BOLD(args.name)} ({args.agent_id})")
    print(f"  Backend : {args.base_url}")
    print(f"  Tick    : {args.tick}s")
    print(f"  Goal    : {CYAN(args.goal)}")
    if args.ticks:
        print(f"  Limit   : {args.ticks} tick(s)")
    print()

    world  = WorldAPI(args.base_url)
    memory = AgentMemory(args.agent_id)
    client = anthropic.Anthropic(api_key=args.api_key)

    # Check backend is alive
    try:
        h = world.health()
        log(GREEN("READY "), f"Backend OK — {h['blocks']} blocks, {h['agents']} agent(s)")
    except Exception as e:
        print(RED(f"ERROR: Cannot reach backend at {args.base_url}"))
        print(f"  {e}")
        print("  Start the backend first:  uvicorn backend.main:app --port 8420")
        sys.exit(1)

    # Register this agent in the world
    agent_pos = {"x": args.start_x, "y": args.start_y, "z": args.start_z}
    world.register_agent(
        args.agent_id, args.name,
        agent_pos["x"], agent_pos["y"], agent_pos["z"],
        args.color,
    )
    log(GREEN("AGENT "), f"Registered '{args.name}' at {tuple(agent_pos.values())}")
    print()

    tick_num = 0
    try:
        while True:
            tick_num += 1
            tick_start = time.time()

            run_tick(client, world, memory, args.agent_id, agent_pos, args.goal, tick_num)

            if args.ticks and tick_num >= args.ticks:
                log(GREEN("DONE  "), f"Reached tick limit ({args.ticks}). Stopping.")
                break

            elapsed = time.time() - tick_start
            wait    = max(0, args.tick - elapsed)
            if wait > 0:
                log(GREY("WAIT  "), f"next tick in {wait:.1f}s  (tick took {elapsed:.1f}s)")
                print()
                time.sleep(wait)

    except KeyboardInterrupt:
        print()
        log(YELLOW("STOP  "), "Interrupted by user. Memory saved.")


if __name__ == "__main__":
    main()