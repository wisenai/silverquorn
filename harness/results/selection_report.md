# Item selection report (v2 — generational-contrast curation, Jul 26)

Source: popquorn_politeness_raw.csv (25042 annotations, 3718 items, 506 annotators). Seed 20260726 (stratified fill only).
Age brackets observed (verbatim): ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '54-59', '60-64', '>65', 'Prefer not to disclose'].
Data-support ceiling: open-ended '>65' bracket (3375 annotations).

Method: curate_items.py scored all items on empirical age signal (old-vs-young mean diff, bracket Spearman) x generational register markers; top-150 (30/quintile) read manually; 40 hand-picked for plausible generational politeness divergence, 8 per quintile, content-filtered. Register pairs included: (386, 2912), (1698, 3179).

items_40 per-quintile counts: {1: 8, 2: 8, 3: 8, 4: 8, 5: 8}
anchors: polite=3643 (mean 5.00), rude=2612 (mean 1.00)

## Curated 40 with rationale tags
- 1729 (q1, mean 2.1): parental-authority teasing; old-young diff +3.0
- 2205 (q1, mean 2.3): joking insult softened with 'j/k' — banter vs. rudeness
- 2278 (q1, mean 2.1): passive-aggressive trailing ellipsis
- 2303 (q1, mean 2.4): pure initialisms (FYI. ILY. SYT. TGIF.)
- 547 (q1, mean 2.3): direct prying personal question
- 712 (q1, mean 2.0): lowercase self-deprecating slang ('i'm a brat!')
- 864 (q1, mean 2.5): 'LOL' + self-deprecation
- 1698 (q1, mean 2.3): casual lowercase apology w/ typo (formal twin: 3179)
- 386 (q2, mean 2.7): formal register version of 2912 (lion's heart pair)
- 679 (q2, mean 2.6): cold formal rebuke — formality as distance
- 1273 (q2, mean 2.6): 'the dude' + all-lowercase casual plan
- 1277 (q2, mean 3.1): curt one-line request, no softening
- 1340 (q2, mean 2.7): playful protest slang ('Don't shush me man')
- 1643 (q2, mean 2.9): 'FYI' + 'addy' abbreviation
- 2232 (q2, mean 2.7): dropped-copula texting style ('We drinking beer tonight?')
- 3424 (q2, mean 2.7): rambling chatty 'Hey guys… Anyways' register
- 127 (q3, mean 3.5): formal wrapper (Dear/Sincerely) on casual content
- 1424 (q3, mean 3.7): terse lowercase vacation note, initials sign-off
- 1579 (q3, mean 3.3): 'pls' abbreviation + initialism sign-off
- 1672 (q3, mean 3.3): doubled question marks ('??')
- 2168 (q3, mean 3.3): winking emoticon + guilt-tease
- 2912 (q3, mean 3.3): texting register version of 386 (lion's heart pair)
- 3004 (q3, mean 3.4): ALL-CAPS-as-default friendly letter
- 3703 (q3, mean 3.7): 'need ASAP' demand wrapped in thanks
- 30 (q4, mean 3.9): no-pleasantries direct work question; diff +2.5
- 507 (q4, mean 3.9): all-lowercase lunch invite
- 513 (q4, mean 3.9): ALL-CAPS heartfelt affection + '!!!!!'
- 1557 (q4, mean 4.1): effusive casual praise ('super job!… awesome')
- 2644 (q4, mean 3.9): old-style spaced emoticon ': )'
- 3015 (q4, mean 4.0): 'you guys' + unpunctuated sign-off
- 3179 (q4, mean 3.7): formal twin of 1698 (superstitious pair)
- 3325 (q4, mean 3.8): mock-humble formality for a bar invite
- 205 (q5, mean 4.3): 'Dear Sir' full formality
- 1141 (q5, mean 4.4): exclamation-dense warmth
- 1751 (q5, mean 4.3): formal apology letter; elders rated it lower (-2.0)
- 1961 (q5, mean 4.4): dated honorific 'madam'
- 2748 (q5, mean 4.4): over-hedged apology ('I hope this does not bother you')
- 3023 (q5, mean 4.3): presumptive courtesy ('Thank you for prioritizing this')
- 3417 (q5, mean 4.5): stilted hyper-formality ('I would kindly like to share')
- 3612 (q5, mean 4.3): 'Thank you in advance' + 'at your earliest convenience'

## Interpretive caveat (for the paper)
Items were SELECTED for hypothesized generational contrast; pilot results therefore speak to whether elder judgments diverge on divergence-prone items, not on a random POPQUORN sample. Say this explicitly in Methods.
