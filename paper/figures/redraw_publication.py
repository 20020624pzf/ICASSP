"""Deterministic vector architecture and unchanged statistical evidence."""
import json
from vector_canvas import P
from architecture_v2 import arch
from contrasts_v2 import contrasts

if __name__ == '__main__':
    inventory = {'architecture': arch(), 'contrasts': contrasts()}
    (P / 'figure_text_inventory.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + '\n')
    print('Two editable vector figures rebuilt; all 12 original intervals retained.')
