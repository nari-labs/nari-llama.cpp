"""Compare bounded test-trellis-model captures using complete FP32 vocabulary logits."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def compare(reference, candidate, max_kl, min_top1):
    a = json.loads(Path(str(reference) + '.json').read_text())
    b = json.loads(Path(str(candidate) + '.json').read_text())
    for key in ('prompt_ids', 'evaluated_generation_ids', 'vocab_size', 'logit_rows'):
        if a[key] != b[key]:
            raise ValueError(f'mismatched capture field: {key}')
    if not a['complete'] or not b['complete'] or a['logit_rows'] < 1:
        raise ValueError('incomplete capture')
    vocab, rows = a['vocab_size'], a['logit_rows']
    size = vocab * rows * 4
    if Path(str(reference) + '.f32').stat().st_size != size or Path(str(candidate) + '.f32').stat().st_size != size:
        raise ValueError('truncated or oversized logits file')
    kls, errors, relative, matches = [], [], [], 0
    with Path(str(reference) + '.f32').open('rb') as af, Path(str(candidate) + '.f32').open('rb') as bf:
        for _ in range(rows):
            x, y = array.array('f'), array.array('f')
            x.fromfile(af, vocab)
            y.fromfile(bf, vocab)
            if not all(math.isfinite(v) for v in x) or not all(math.isfinite(v) for v in y):
                raise ValueError('nonfinite logits')
            mx, my = max(x), max(y)
            lse_x = mx + math.log(math.fsum(math.exp(v - mx) for v in x))
            lse_y = my + math.log(math.fsum(math.exp(v - my) for v in y))
            kl = math.fsum(math.exp(v - lse_x) * ((v - lse_x) - (w - lse_y)) for v, w in zip(x, y))
            kls.append(max(0.0, kl))
            errors.append(max(abs(v - w) for v, w in zip(x, y)))
            relative.append(math.sqrt(math.fsum((v - w) ** 2 for v, w in zip(x, y)) / max(1e-30, math.fsum(v * v for v in x))))
            matches += x.index(mx) == y.index(my)
    report = {
        'complete': True, 'rows': rows, 'vocab_size': vocab,
        'mean_kl': math.fsum(kls) / rows, 'max_kl': max(kls), 'per_row_kl': kls,
        'top1_agreement': matches / rows, 'max_abs_logit_error': max(errors),
        'max_relative_l2': max(relative), 'greedy_ids_equal': a['greedy_ids'] == b['greedy_ids'],
        'reference_logits_sha256': digest(Path(str(reference) + '.f32')),
        'candidate_logits_sha256': digest(Path(str(candidate) + '.f32')),
        'thresholds': {'max_kl': max_kl, 'min_top1': min_top1},
    }
    report['passed'] = (report['max_kl'] <= max_kl and report['top1_agreement'] >= min_top1
                        and report['greedy_ids_equal'])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path, help='capture prefix without extension')
    p.add_argument('candidate', type=Path, help='capture prefix without extension')
    p.add_argument('--max-kl', type=float, default=1e-3)
    p.add_argument('--min-top1', type=float, default=1.0)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.max_kl < 0 or not 0 <= args.min_top1 <= 1:
        p.error('invalid comparison thresholds')
    report = compare(args.reference, args.candidate, args.max_kl, args.min_top1)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
