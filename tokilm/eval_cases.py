"""Manual prompts for spot-checking generations and translations."""

EVAL_CASES = [
    "toki",
    "sina pona",
    "mi moku",
    "jan li toki",
    "tenpo suno la",
    "English: I am happy.\nToki Pona:",
    "Toki Pona: sina pona.\nEnglish:",
]


def get_eval_cases():
    return list(EVAL_CASES)
