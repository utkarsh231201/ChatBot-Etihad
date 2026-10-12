"""Deterministic guardrails (input + output). LLM-side guardrails live in prompts.py."""
import re

APPROVAL_REQUEST_RE = re.compile(
    r"\b(approve|approved|grant|process|issue|authori[sz]e|waive|override|expedite|confirm)\b"
    r".{0,60}\b(my\s+|a\s+|the\s+)?(refund|waiver|compensation|fee waiver|exception)\b|"
    r"\b(refund|waiver|compensation)\b.{0,40}\b(for me|me now|right now|immediately|on my booking)\b",
    re.IGNORECASE)

# Phrases the assistant must never emit: they would amount to approving something.
APPROVAL_OUTPUT_RE = re.compile(
    r"(i (have |'ve |will |can )?(now )?(approved?|waived?|granted|processed|issued)\b"
    r"|(your|the) (refund|waiver|compensation) (has been|is|was) (approved|granted|processed|issued|waived)"
    r"|(refund|waiver) (approved|granted)|i('ll| will) (refund|waive))", re.IGNORECASE)

REFUSAL_APPROVAL = (
    "I can't approve, process or promise refunds, waivers or exceptions - those decisions are made "
    "by Etihad staff on a case-by-case basis. I can explain what the published policy says, or you "
    "can contact Etihad Guest Services / your booking channel to request a review.")

REFUSAL_NO_CONTEXT = (
    "I couldn't find this in the Etihad policy documents I have access to, so I can't answer it "
    "reliably. Please contact Etihad Guest Services for confirmation, or rephrase your question.")

REFUSAL_UNSUPPORTED = (
    "I can only help with Etihad Airways policies (baggage, pets, travel documents, booking rules, "
    "etc.) and flight status, schedules and routes. I can't help with that request.")

REFUSAL_UNGROUNDED = (
    "I wasn't able to produce an answer that is fully supported by the policy documents, so I'd "
    "rather not guess. Please contact Etihad Guest Services for an authoritative answer.")

SAFETY_NO_INFO = (
    "My policy documents don't cover dangerous goods such as lighters, perfumes, aerosols, "
    "batteries or other flammable items, so I can't tell you what is allowed. Items like these "
    "can be restricted or banned on aircraft because they can be dangerous to passengers and "
    "crew. Please don't pack them until you have confirmed the rules: check Etihad's prohibited "
    "items and dangerous goods information, or contact Etihad Guest Services before you travel.")

SUPPORT_MESSAGE = (
    "I'm not sure what you mean, and I want to check in first. If you are having thoughts of "
    "hurting yourself, you're not alone and help is available. If you are in immediate danger, "
    "contact your local emergency number now, and if you are on a flight, tell a member of the "
    "cabin crew. In the US you can call or text 988, and findahelpline.com lists crisis lines "
    "in other countries. If you meant something else, such as changing, skipping or rebooking a "
    "flight, tell me more and I'll help with Etihad's policies.")

REFUSAL_FARES = (
    "I can't search fares, compare prices, check seat availability or book tickets, so I can't "
    "tell you what a flight costs or which one is cheapest. My flight data covers status, schedules and routes only. For "
    "current prices and offers, please check etihad.com or the Etihad app. I can show which "
    "Etihad flights operate between two airports (for example, ask \"flights from Mumbai to "
    "Abu Dhabi\"), and I can explain Etihad's policies and published fees.")

REFUSAL_HARMFUL = (
    "I can't help with that. I won't give advice on getting items past security screening or on "
    "anything that could put passengers, crew or an aircraft at risk. If you have a genuine "
    "question about what you may carry, ask me and I'll point you to the official rules.")

TOOL_ERROR =("I couldn't retrieve live flight data right now. Please check the Etihad website or "
              "app for the latest information.")

GREETING = ("Hello! I can answer questions about Etihad policies and look up flight status, "
            "schedules and routes. What would you like to know?")


# Questions about carrying hazardous items. We still try the documents first; this only decides
# which message is shown when the documents have no answer (never guess about safety).
SAFETY_TOPIC_RE = re.compile(
    r"\b(lighters?|matches|matchbox|perfumes?|colou?gnes?|aerosols?|sprays?|flammables?|inflammable|"
    r"combustible|fuel|petrol|gasoline|butane|propane|gas canisters?|lithium|power ?banks?|"
    r"batter(y|ies)|explosives?|fireworks?|ammunition|firearms?|guns?|weapons?|knife|knives|"
    r"blades?|pepper spray|chemicals?|bleach|dangerous goods|hazardous|corrosive|radioactive|"
    r"toxic|poison|dry ice|oxygen|e-?cigarettes?|vapes?|nail polish|alcohol)\b", re.IGNORECASE)

# Requests to evade screening or to endanger people. Deliberately narrow: "is it harmful to carry
# a lighter?" is a legitimate safety question and must NOT match.
HARMFUL_RE = re.compile(
    r"\b(smuggl\w*|sneak\w*|hide|hiding|conceal\w*|bypass\w*|evade|evading|avoid\w*|get(ting)? past|"
    r"trick\w*|fool\w*)\b.{0,60}\b(security|screening|scanners?|x-?rays?|customs|detection|"
    r"checkpoints?|inspection|airport staff)\b|"
    r"\bsmuggl\w*\b|"
    r"\b(hijack\w*|sabotag\w*|blow up|bomb threat)\b|"
    r"\b(hurt|kill|injure|poison|attack)\w*\b.{0,40}\b(passengers?|crew|pilots?|flight|aircraft|plane)\b|"
    r"\bhow (do|can|to)\b.{0,30}\b(make|build|assemble)\b.{0,20}\b(bomb|explosive|weapon)s?\b",
    re.IGNORECASE)


# Possible self-harm. Checked BEFORE the harmful-request rule: these people need care, not a
# security refusal. Wording is deliberately broad because the reply is a gentle check-in that
# also offers a normal-flight reading, so a false match costs little and a miss could cost a lot.
SELF_HARM_RE = re.compile(
    r"\b(jump|jumping|leap|leaping|throw myself|throwing myself|fall|falling)\b.{0,30}"
    r"\b(out of|off|from)\s+(a\s+|an\s+|the\s+|my\s+|this\s+)?(moving\s+|flying\s+)?"
    r"(plane|aircraft|airplane|aeroplane|flight|window|door)\b(?!\s+to\b)|"
    r"\b(kill|hurt|harm|end)\s+(myself|my life|my own life)\b|"
    r"\bsuicid\w*|\bwant to die\b|\bself[- ]?harm\b|\bend it all\b|\bdon'?t want to (live|be alive)\b",
    re.IGNORECASE)


def is_self_harm_risk(text: str) -> bool:
    return bool(SELF_HARM_RE.search(text))


def is_safety_topic(text: str) -> bool:
    return bool(SAFETY_TOPIC_RE.search(text))


def is_harmful_request(text: str) -> bool:
    return bool(HARMFUL_RE.search(text))


def no_info_message(query: str, default: str) -> str:
    """Pick the 'not in my documents' reply: safety wording for hazardous-item questions."""
    return SAFETY_NO_INFO if is_safety_topic(query) else default


def is_approval_request(text: str) -> bool:
    return bool(APPROVAL_REQUEST_RE.search(text))


def violates_output_policy(text: str) -> bool:
    return bool(APPROVAL_OUTPUT_RE.search(text))
