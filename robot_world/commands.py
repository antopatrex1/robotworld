"""Deterministic local language commands; no cloud key or arbitrary code execution."""
import re
from dataclasses import dataclass
from .objects import OBJECTS

@dataclass(frozen=True)
class Command:
    action: str
    values: tuple = ()


def parse_command(text):
    text=re.sub(r'\s+',' ',text.strip().lower()).rstrip('!?.,').strip()
    # Keep the action grammar bounded while accepting normal spoken requests.
    # These wrappers add no actions; the remaining command must still match in full.
    while True:
        unwrapped=re.sub(r'^(?:please|robot,?|(?:can|could|would|will) you)(?: |$)','',text)
        if unwrapped==text: break
        text=unwrapped
    text=re.sub(r',? please$','',text).strip()
    if not text: raise ValueError('Type a command, then press Enter or Run command.')
    if text in ('stop','pause','cancel'): return Command('stop')
    if text in ('reset','start over'): return Command('reset')
    if text in ('walk around the table','walk around table','circle the table'): return Command('tour')
    if re.fullmatch(r'(open|close) (the )?(hands|fingers|both hands)',text):
        return Command('hands',(0. if text.startswith('open') else 1.,))
    m=re.fullmatch(r'(?:walk|move|go) (forward|backward|backwards|back|left|right)(?: (\d+(?:\.\d+)?)\s*(?:m|meters?|metres?))?',text)
    if m:
        distance=float(m[2] or 1)
        if not .1<=distance<=5: raise ValueError('Walking distance must be between 0.1 and 5 metres.')
        return Command('walk',(m[1],distance))
    m=re.fullmatch(r'(?:walk|go|move) to x\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*y\s*(-?\d+(?:\.\d+)?)',text)
    if m: return Command('position',(float(m[1]),float(m[2])))
    m=re.fullmatch(r'turn (left|right)(?: (\d+(?:\.\d+)?) degrees)?',text)
    if m:
        angle=float(m[2] or 90)
        if angle>360: raise ValueError('Choose a turn of 360 degrees or less.')
        return Command('turn',(angle*(1 if m[1]=='left' else -1),))
    for key,info in OBJECTS.items():
        for alias in sorted(info['aliases'],key=len,reverse=True):
            if text in (f'go to {alias}',f'go to the {alias}',f'walk to {alias}',f'walk to the {alias}'):
                return Command('approach',(key,))
            if text in (f'reach for {alias}',f'reach for the {alias}',f'touch {alias}',f'touch the {alias}'):
                return Command('reach',(key,))
            if re.fullmatch(r'(?:pick up|grab|grasp) (?:the |a |an )?'+re.escape(alias),text) or text in (
                f'pick {alias} up',f'pick the {alias} up',f'pick up and move the {alias}'):
                return Command('pick',(key,))
    raise ValueError('Try: walk backward 1 meter; walk around the table; go to the red mug; reach for the blue bottle; pick up the blue bottle; open hands; stop.')
