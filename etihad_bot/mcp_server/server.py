"""MCP server exposing flight tools over stdio. Run: python -m etihad_bot.mcp_server.server"""
import json

from mcp.server.fastmcp import FastMCP

from etihad_bot.mcp_server import flight_api

mcp = FastMCP("etihad-flights")


def _safe(fn, *a, **kw) -> str:
    try:
        return json.dumps({"ok": True, **fn(*a, **kw)})  # {"results": [...], "notes": [...]}
    except flight_api.FlightAPIError as e:
        return json.dumps({"ok": False, "error": str(e)})
    except Exception as e:  # network errors etc.
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})


@mcp.tool()
def get_flight_status(flight_number: str, date: str = "") -> str:
    """Live status of a flight (e.g. EY101): scheduled/estimated/actual times, gate, terminal, delay.
    `date` is optional, YYYY-MM-DD."""
    return _safe(flight_api.flight_status, flight_number, date or None)


@mcp.tool()
def get_flight_schedule(origin: str, destination: str, date: str = "") -> str:
    """Etihad flights between two airports (IATA codes, e.g. AUH -> LHR) with scheduled times.
    `date` is optional, YYYY-MM-DD."""
    return _safe(flight_api.flight_schedule, origin, destination, date or None)


@mcp.tool()
def get_flight_routes(origin: str = "", destination: str = "") -> str:
    """Etihad operated routes from and/or to an airport (IATA codes). At least one is required."""
    if not origin and not destination:
        return json.dumps({"ok": False, "error": "origin or destination is required"})
    return _safe(flight_api.flight_routes, origin or None, destination or None)


if __name__ == "__main__":
    mcp.run(transport="stdio")
