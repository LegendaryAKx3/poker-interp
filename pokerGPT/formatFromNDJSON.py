def format_poker_sequence(actions):
    return " | ".join(actions)

default = [
    "d dh p1 2hAc", "d dh p2 8hQs", "d dh p3 3cKc", "d dh p4 8c7s",
    "d dh p5 7dAs", "d dh p6 Th8s", "p3 f", "p4 f", "p5 cc", "p6 cc",
    "p1 cc", "p2 cc", "d db 6dTs7h", "p1 cc", "p2 cc", "p5 cc",
    "d db 9h", "p1 cc", "p2 cc", "p5 cc", "d db 9d", "p1 cc",
    "p2 cbr 640", "p5 cc", "p6 cbr 2795", "p1 f", "p2 cbr 9800",
    "p5 f", "p6 cc", "p2 sm 8hQs", "p6 sm Th8s"
]

formatted = format_poker_sequence(default)
print(formatted)
