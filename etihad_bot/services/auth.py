"""PNR sign-in. A PNR has the exact shape LL-DD-LL: two uppercase letters, two digits, two
uppercase letters (e.g. 'AB23FG'). It is case-sensitive: 'ab23fg' is rejected, never silently
upper-cased, because 'ab23fg' and 'AB23FG' are not the same reference.

NOTE: there is no booking system behind this, so a PNR is *not verified* against Etihad - it is
just an identity key. Anyone who knows a PNR can sign in as it.
"""
import hashlib
import re

from etihad_bot.config import get_settings

PNR_RE = re.compile(r"[A-Z]{2}[0-9]{2}[A-Z]{2}")

PNR_FORMAT_HELP = ("A PNR must be 2 uppercase letters, then 2 digits, then 2 uppercase letters, "
                   "for example 'AB23FG'. It is case-sensitive: 'ab23fg' is not accepted.")


def validate_pnr(pnr: str) -> str | None:
    """Return an error message, or None if the PNR is well-formed."""
    if not PNR_RE.fullmatch(pnr):
        return PNR_FORMAT_HELP
    return None


def user_id_for_pnr(pnr: str) -> str:
    """Stable pseudonymous user id so the raw PNR is not written to logs or tables."""
    digest = hashlib.sha256(f"{get_settings().auth_pepper}:{pnr}".encode()).hexdigest()[:16]
    return f"pnr:{digest}"
