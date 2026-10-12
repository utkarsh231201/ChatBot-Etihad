"""Third-party flight data provider (AviationStack). Falls back to mock data if FLIGHT_API_MOCK=true."""
import os
from datetime import datetime

import requests

BASE = "http://api.aviationstack.com/v1"  # free tier is HTTP only


class FlightAPIError(RuntimeError):
    pass


def _mock() -> bool:
    return os.getenv("FLIGHT_API_MOCK", "false").lower() == "true"


def _get(endpoint: str, params: dict) -> list[dict]:
    key = os.getenv("AVIATIONSTACK_API_KEY", "")
    if not key:
        raise FlightAPIError("AVIATIONSTACK_API_KEY is not configured")
    params = {k: v for k, v in params.items() if v} | {"access_key": key}
    r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=20)
    data = r.json()
    if "error" in data:
        raise FlightAPIError(data["error"].get("message", "provider error"))
    return data.get("data", [])


def _slim_flight(f: dict) -> dict:
    dep, arr = f.get("departure") or {}, f.get("arrival") or {}
    return {
        "flight": (f.get("flight") or {}).get("iata"),
        "airline": (f.get("airline") or {}).get("name"),
        "status": f.get("flight_status"),
        "date": f.get("flight_date"),
        "departure": {k: dep.get(k) for k in ("airport", "iata", "terminal", "gate", "scheduled",
                                               "estimated", "actual", "delay")},
        "arrival": {k: arr.get(k) for k in ("airport", "iata", "terminal", "gate", "scheduled",
                                             "estimated", "actual", "delay")},
    }


_MOCK_FLIGHT = {
    "flight": "EY101", "airline": "Etihad Airways", "status": "scheduled", "date": "2026-01-01",
    "departure": {"airport": "Abu Dhabi International", "iata": "AUH", "terminal": "A",
                  "gate": "A12", "scheduled": "2026-01-01T09:35:00+00:00", "estimated": None,
                  "actual": None, "delay": None},
    "arrival": {"airport": "Heathrow", "iata": "LHR", "terminal": "4", "gate": None,
                "scheduled": "2026-01-01T13:50:00+00:00", "estimated": None, "actual": None,
                "delay": None},
}


NO_RESULTS_NOTE = ("No Etihad (EY) flights found in the provider's live feed for this request. The feed "
                   "only covers currently scheduled/active flights, so this does not prove no such "
                   "flight exists.")


def _iso_date(d: str | None) -> str | None:
    """Accept YYYY-MM-DD or day-first DD/MM/YYYY; anything else is ignored."""
    if not d:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(d.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _flights(params: dict, date: str | None, notes: list[str]) -> list[dict]:
    """`flight_date` is a paid-plan feature: if refused, retry without it and say so."""
    iso = _iso_date(date)
    if date and not iso:
        notes.append(f"Could not read the date '{date}'; it was ignored.")
    if iso:
        try:
            return _get("flights", {**params, "flight_date": iso})
        except FlightAPIError as e:
            if "subscription plan" not in str(e):
                raise
            notes.append(f"Filtering by date ({iso}) is not available on the current data plan; "
                         "these are currently tracked flights and may not be for that date.")
    return _get("flights", params)


def flight_status(flight_iata: str, flight_date: str | None = None) -> dict:
    if _mock():
        return {"results": [{**_MOCK_FLIGHT, "flight": flight_iata.upper()}]}
    notes: list[str] = []
    flights = [_slim_flight(f) for f in _flights({"flight_iata": flight_iata.upper()}, flight_date, notes)]
    exact = [f for f in flights if (f["flight"] or "").upper() == flight_iata.upper()]
    if exact:  # drop codeshare duplicates when the exact flight number exists
        flights = exact
    elif flights:
        notes.append(f"No record listed under {flight_iata.upper()}; these are the provider's matches "
                     "for that flight number (likely codeshare listings under another number).")
    else:
        notes.append(f"No live record found for {flight_iata.upper()}.")
    return {"results": flights[:5], "notes": notes}


def flight_schedule(dep_iata: str, arr_iata: str, flight_date: str | None = None) -> dict:
    if _mock():
        return {"results": [_MOCK_FLIGHT | {
            "departure": _MOCK_FLIGHT["departure"] | {"iata": dep_iata.upper()},
            "arrival": _MOCK_FLIGHT["arrival"] | {"iata": arr_iata.upper()}}]}
    notes: list[str] = []
    rows = [_slim_flight(f) for f in _flights(
        {"dep_iata": dep_iata.upper(), "arr_iata": arr_iata.upper(), "airline_iata": "EY"},
        flight_date, notes)]
    if not rows:
        notes.append(NO_RESULTS_NOTE)
    return {"results": rows[:10], "notes": notes}


def _routes_from_flights(dep_iata: str | None, arr_iata: str | None) -> dict:
    """Free-plan fallback: infer Etihad routes (with flight numbers) from currently tracked EY flights."""
    routes: dict[tuple, set] = {}
    for f in _get("flights", {"airline_iata": "EY", "dep_iata": (dep_iata or "").upper(),
                              "arr_iata": (arr_iata or "").upper(), "limit": 100}):
        sf = _slim_flight(f)
        routes.setdefault((sf["departure"]["iata"], sf["arrival"]["iata"]), set()).add(sf["flight"])
    out = [{"departure_iata": d, "arrival_iata": a, "flights": sorted(n)}
           for (d, a), n in sorted(routes.items())][:25]
    notes = ["Routes were inferred from currently tracked Etihad flights, so the list may be incomplete."]
    if not out:
        notes.append(NO_RESULTS_NOTE)
    return {"results": out, "notes": notes}


def flight_routes(dep_iata: str | None = None, arr_iata: str | None = None) -> dict:
    if _mock():
        return {"results": [{"flight": "EY101", "departure_iata": dep_iata or "AUH",
                             "arrival_iata": arr_iata or "LHR", "days": ["mon", "wed", "fri"]}]}
    try:
        rows = _get("routes", {"airline_iata": "EY", "dep_iata": (dep_iata or "").upper(),
                               "arr_iata": (arr_iata or "").upper()})[:15]
    except FlightAPIError as e:
        if "subscription plan" not in str(e):
            raise
        return _routes_from_flights(dep_iata, arr_iata)  # `routes` needs a paid plan
    return {"results": [{"flight": (r.get("flight") or {}).get("iata"),
                         "departure_iata": (r.get("departure") or {}).get("iata"),
                         "arrival_iata": (r.get("arrival") or {}).get("iata"),
                         "departure_time": (r.get("departure") or {}).get("time"),
                         "arrival_time": (r.get("arrival") or {}).get("time")} for r in rows]}

