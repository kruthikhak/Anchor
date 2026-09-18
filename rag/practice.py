import random
import re

from . import prompts
from .llm import chat_json

BLANK = re.compile(r"_{2,}")  # models write the gap as __ or as a long run of underscores
LETTERS = "ABCDE"

# how a quiz is made up: a short topic gets five marks, a long one ten
QUIZ_MIX = {5: {"mcq": 3, "blank": 1, "short": 1}, 10: {"mcq": 6, "blank": 2, "short": 2}}
LONG_SECTION = 10  # passages in the book's section, where the deadlock section has 15 and ACID's 4


def source_number(value, count):
    # the model sometimes writes the source as "2" or [2] instead of 2
    if isinstance(value, list):
        value = value[0] if value else None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= count else None


def text(value):
    return value.strip() if isinstance(value, str) else ""


def practice_questions(topic, sources):
    """Three to five recall questions, each with its answer and a short explanation."""
    reply = chat_json(
        [
            {"role": "system", "content": prompts.PRACTICE_SYSTEM},
            {"role": "user", "content": prompts.answer_request(topic, sources)},
        ],
        temperature=0.4,
        reasoning_effort="low",
        max_completion_tokens=2000,
    )
    questions = reply.get("questions") if isinstance(reply, dict) else None
    kept = []
    # a question that can't point at one of the sources wasn't written from them, so it's dropped
    for q in questions if isinstance(questions, list) else []:
        number = source_number(q.get("source"), len(sources)) if isinstance(q, dict) else None
        if number and text(q.get("question")) and text(q.get("answer")):
            kept.append({"question": text(q["question"]), "answer": text(q["answer"]),
                         "explanation": text(q.get("explanation")), "source": number})
    return kept


def quiz_marks(sources, chunks):
    """Ten marks when the section the best passage comes from is a long one in its book."""
    top = sources[0].hit.chunk
    section = " > ".join(p.strip() for p in top.section.split(">")[:2])
    length = sum(1 for c in chunks if c.doc_id == top.doc_id and (c.section == section or c.section.startswith(section + " > ")))
    return 10 if length >= LONG_SECTION else 5


def quiz_questions(topic, sources, marks):
    mix = QUIZ_MIX[marks]
    request = prompts.answer_request(topic, sources) + "\n\n" + prompts.quiz_instruction(**mix)
    reply = chat_json(
        [{"role": "system", "content": prompts.QUIZ_SYSTEM}, {"role": "user", "content": request}],
        temperature=0.4,
        reasoning_effort="low",
        max_completion_tokens=3500,
    )
    questions = reply.get("questions") if isinstance(reply, dict) else None
    checked = (check_question(q, len(sources)) for q in (questions if isinstance(questions, list) else []))
    return [q for q in checked if q][:marks]


def check_question(q, count):
    """The question as the page needs it, or None when it doesn't hold up."""
    if not isinstance(q, dict):
        return None
    number, question = source_number(q.get("source"), count), text(q.get("question"))
    if not number or not question:
        return None
    base = {"question": question, "explanation": text(q.get("explanation")), "source": number}

    if q.get("type") == "mcq":
        options = q.get("options")
        if isinstance(options, dict):  # {"A": "...", "B": "..."}
            options = [options[k] for k in sorted(options)]
        if not isinstance(options, list) or not 3 <= len(options) <= 5 or not all(text(o) for o in options):
            return None
        options = [text(o) for o in options]
        if len({o.lower() for o in options}) < len(options):
            return None
        answer = q.get("answer")
        if isinstance(answer, list) and len(answer) == 1:
            answer = answer[0]
        if isinstance(answer, str):
            # the position as text, a letter, or the option itself all turn up
            answer = answer.strip().rstrip(".)")
            lowered = [o.lower() for o in options]
            if answer.isdigit():
                answer = int(answer)
            elif len(answer) == 1 and answer.upper() in LETTERS:
                answer = LETTERS.index(answer.upper())
            elif answer.lower() in lowered:
                answer = lowered.index(answer.lower())
        if not isinstance(answer, int) or isinstance(answer, bool) or not 0 <= answer < len(options):
            return None
        # models like to put the right answer first, so the options are shuffled
        order = list(range(len(options)))
        random.Random(question).shuffle(order)
        return {**base, "type": "mcq", "options": [options[i] for i in order], "answer": order.index(answer)}

    if q.get("type") == "blank":
        answer = text(q.get("answer"))
        if len(BLANK.findall(question)) != 1 or not answer or len(answer.split()) > 6:
            return None
        accept = [text(a) for a in q.get("accept", []) if text(a)] if isinstance(q.get("accept"), list) else []
        return {**base, "type": "blank", "question": BLANK.sub("____", question), "answer": answer, "accept": accept}

    if q.get("type") == "short" and text(q.get("answer")):
        return {**base, "type": "short", "answer": text(q["answer"])}
    return None
