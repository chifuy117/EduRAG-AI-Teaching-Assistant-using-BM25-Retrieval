"""Score analysis — reads eval_report.json and decides if retraining is needed."""
import json, sys
sys.path.insert(0, '.')

r = json.load(open('data/eval_report.json'))
ret = r['retrieval']
cov = r['data_coverage']
cls = r['classification']
noise = r['noise_analysis']

f1        = ret.get('macro_f1', 0)
precision = ret.get('macro_precision', 0)
recall    = ret.get('macro_recall', 0)
MAP       = ret.get('MAP', 0)
ndcg      = ret.get('mean_ndcg', 0)
mrr       = ret.get('MRR', 0)
subj_acc  = ret.get('subject_accuracy', 0)
acc       = cls.get('overall_accuracy', 0)
macro_f1c = cls.get('macro_f1', 0)
clean     = noise.get('clean_pct', 0)
chunks    = cov.get('total_chunks', 0)

print('\n' + '='*55)
print('  EVALUATION SCORE ANALYSIS')
print('='*55)
print(f'  Retrieval F1 (macro):     {f1:.4f}   {"OK" if f1 >= 0.8 else "LOW"}')
print(f'  Precision@5 (macro):      {precision:.4f}   {"OK" if precision >= 0.7 else "LOW"}')
print(f'  Recall@5    (macro):      {recall:.4f}')
print(f'  MAP  (mean avg prec):     {MAP:.4f}   {"OK" if MAP >= 0.7 else "LOW"}')
print(f'  MRR  (mean recip rank):   {mrr:.4f}')
print(f'  Mean NDCG@5:              {ndcg:.4f}   {"EXCELLENT" if ndcg >= 0.9 else "GOOD"}')
print(f'  Subject Hit Accuracy:     {subj_acc:.4f}   {"PERFECT" if subj_acc >= 1.0 else "OK"}')
print(f'  Topic Classification Acc: {acc:.4f}   {"OK" if acc >= 0.6 else "LOW"}')
print(f'  Macro F1 (classification):{macro_f1c:.4f}')
print(f'  Chunk Clean %:            {clean}%      {"OK" if clean >= 80 else "LOW"}')
print(f'  Total Chunks:             {chunks}')
print()
print('  By Subject:')
for subj, cnt in cov.get('by_subject', {}).items():
    bar = chr(9608) * (cnt // 10)
    print(f'    {subj:20s}  {cnt:4d}  {bar}')
print()

issues = []
if f1 < 0.80:       issues.append('Retrieval F1 below 0.80')
if precision < 0.70: issues.append('Precision below 0.70')
if acc < 0.50:      issues.append('Classification accuracy below 0.50')
if clean < 80:      issues.append('Clean chunk % below 80')
if chunks < 100:    issues.append('Too few chunks (< 100)')
if MAP < 0.70:      issues.append('MAP below 0.70')

RETRAIN = len(issues) > 0

print('='*55)
if RETRAIN:
    print('  DECISION: RETRAIN RECOMMENDED')
    print('  Issues:')
    for iss in issues:
        print(f'    - {iss}')
else:
    print('  DECISION: NO RETRAINING NEEDED')
    print('  All metrics meet or exceed thresholds.')
print('='*55 + '\n')
