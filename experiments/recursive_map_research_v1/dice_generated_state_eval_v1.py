"""Reuse the registered native Q evaluator with the new matched arm checkpoints."""
from paths import *


def main():
    base = JOB / 'dice_schedule_eval_v1.py'
    source = base.read_text()
    for old, new in (
        ("DEST = JOB / 'dice_schedule_probe_v1'", "DEST = JOB / 'dice_generated_state_v1'"),
        ("choices=['old', 'corrected']", "choices=['baseline', 'generated']"),
        ("read(JOB / 'runs/g2_b' / f'seed{args.seed}/CHECKPOINT_000300.json') if args.update == 0", "read(plan['starts'][str(args.seed)]) if args.update == 0"),
    ):
        assert source.count(old) == 1
        source = source.replace(old, new)
    exec(compile(source, str(base), 'exec'), {'__name__': '__main__', '__file__': str(base)})


if __name__ == '__main__':
    main()
