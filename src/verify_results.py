"""Consistency checks on actual completed results; no accuracy is invented."""
import csv,json,math,hashlib
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from .experiment import ROOT,EXPERIMENTS


def main():
    out=ROOT/'results';split=json.loads((out/'split.json').read_text())
    env=json.loads((out/'environment.json').read_text())
    assert env['source_sha256']==hashlib.sha256((ROOT/'src/experiment.py').read_bytes()).hexdigest()
    a,b=split['train_indices'],split['val_indices']
    assert len(a)==45000 and len(b)==5000 and len(set(a)|set(b))==50000 and not set(a)&set(b)
    validation={}
    for cfg in EXPERIMENTS:
        folder=out/cfg['name'];config=json.loads((folder/'config.json').read_text());s=json.loads((folder/'summary.json').read_text())
        with (folder/'history.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==config['epochs']==5
        assert [int(r['epoch']) for r in rows]==list(range(1,6))
        assert all(math.isfinite(float(v)) for r in rows for v in r.values())
        assert all(0<=float(r[k])<=1 for r in rows for k in ['train_accuracy','val_accuracy'])
        best=max(rows,key=lambda r:float(r['val_accuracy']))
        assert int(best['epoch'])==s['best_epoch'] and float(best['val_accuracy'])==s['best_val_accuracy']
        event=EventAccumulator(str(folder/'tensorboard')).Reload()
        for tag in ['train_loss','val_loss','train_accuracy','val_accuracy']:
            vals=event.Scalars(tag);assert len(vals)==5
            for record,row in zip(vals,rows):assert record.step==int(row['epoch']) and abs(record.value-float(row[tag]))<1e-5
        validation[cfg['name']]=s['best_val_accuracy']
        target=folder/'test_metrics.json'
        if target.exists():
            t=json.loads(target.read_text());matrix=np.array(t['confusion_matrix'])
            assert t['samples']==10000 and matrix.shape==(10,10) and matrix.sum()==10000
            assert np.all(matrix.sum(axis=1)==1000)
            assert abs(matrix.trace()/10000-t['accuracy'])<1e-12
            assert t['checkpoint_epoch']==s['best_epoch']
            with (folder/'test_predictions.csv').open() as f:assert sum(1 for _ in csv.DictReader(f))==10000
        print(cfg['name'], 'CSV / TensorBoard / validation / test checks OK',flush=True)
    selection=json.loads((out/'selection_before_test.json').read_text())
    assert selection['winner']==max(validation,key=validation.get)
    for name in selection['evaluate_final_models']:assert (out/name/'test_metrics.json').exists()
    proof={'status':'passed','real_experiments':5,'epochs_per_experiment':5,'train_samples':45000,'val_samples':5000,
           'test_samples_per_evaluated_model':10000,'checks':['split disjoint','finite history','best checkpoint by validation',
           'TensorBoard matches CSV','confusion matrices match test accuracy','predictions count','selection by validation','executed source hash']}
    (out/'verification.json').write_text(json.dumps(proof,indent=2))
    print('ALL RESULT CHECKS PASSED')

if __name__=='__main__':main()
