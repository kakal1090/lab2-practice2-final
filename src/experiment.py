"""Complete Practice 2 experiments on full CIFAR-10; real results only."""
import argparse, csv, hashlib, json, os, platform, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import torchvision
from sklearn.metrics import classification_report, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
EXPERIMENTS=[
 {'name':'resnet_baseline','model':'resnet18','mode':'head','lr':.001,'batch_size':32},
 {'name':'densenet_baseline','model':'densenet121','mode':'head','lr':.001,'batch_size':32},
 {'name':'resnet_lr_01','model':'resnet18','mode':'head','lr':.01,'batch_size':32},
 {'name':'resnet_batch64','model':'resnet18','mode':'head','lr':.001,'batch_size':64},
 {'name':'resnet_finetune','model':'resnet18','mode':'finetune','lr':.001,'batch_size':32},
]

def write_json(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def make_model(name, mode='head', pretrained=True):
    if name=='resnet18':
        m=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        for p in m.parameters():p.requires_grad=False
        torch.manual_seed(42);m.fc=nn.Linear(m.fc.in_features,10)
        if mode=='finetune':
            for p in m.layer4.parameters():p.requires_grad=True
    else:
        m=models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        for p in m.parameters():p.requires_grad=False
        torch.manual_seed(42);m.classifier=nn.Linear(m.classifier.in_features,10)
    return m

def head(m,name):return m.fc if name=='resnet18' else m.classifier

def features(m,name,x):
    if name=='resnet18':
        x=m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x=m.layer4(m.layer3(m.layer2(m.layer1(x))))
        return m.avgpool(x).flatten(1)
    return torch.nn.functional.adaptive_avg_pool2d(torch.relu(m.features(x)),(1,1)).flatten(1)

def training_mode(m,name,mode):
    m.eval();head(m,name).train()
    if mode=='finetune':m.layer4.train()

def split_indices(targets):
    rng=random.Random(42);tr=[];va=[]
    for c in range(10):
        idx=[i for i,y in enumerate(targets) if y==c];rng.shuffle(idx)
        va+=idx[:500];tr+=idx[500:]
    rng.shuffle(tr);rng.shuffle(va)
    assert len(tr)==45000 and len(va)==5000 and not set(tr)&set(va)
    return tr,va

def loaders_data(size,workers):
    tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),
        transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    ds=datasets.CIFAR10(str(ROOT/'data'),train=True,download=False,transform=tf)
    tr,va=split_indices(ds.targets)
    return {'train':Subset(ds,tr),'val':Subset(ds,va)},tf,{'seed':42,'train_indices':tr,'val_indices':va}

def extract(m,name,ds,device,batch,workers,label):
    m.eval();xs=[];ys=[];start=time.perf_counter()
    dl=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=workers)
    with torch.inference_mode():
        for i,(x,y) in enumerate(dl,1):
            z=features(m,name,x.to(device));xs.append(z.cpu());ys.append(y)
            if i==1 or i%100==0 or i==len(dl):print(f'{name} features {label}: {i}/{len(dl)}',flush=True)
    return TensorDataset(torch.cat(xs),torch.cat(ys)),time.perf_counter()-start

def run_epoch(m,dl,device,opt=None,label='',predict=False):
    loss_sum=0.;correct=0;n=0;truth=[];pred=[]
    with torch.set_grad_enabled(opt is not None):
        for i,(x,y) in enumerate(dl,1):
            x=x.to(device);y=y.to(device)
            if opt is not None:opt.zero_grad(set_to_none=True)
            z=m(x);loss=nn.functional.cross_entropy(z,y)
            if opt is not None:loss.backward();opt.step()
            n+=len(y);loss_sum+=float(loss.detach())*len(y);p=z.argmax(1);correct+=int((p==y).sum())
            if predict:truth.extend(y.cpu().tolist());pred.extend(p.cpu().tolist())
            if i==1 or i%300==0 or i==len(dl):print(f'{label}: batch {i}/{len(dl)} loss={loss_sum/n:.4f} acc={correct/n:.2%}',flush=True)
    return {'loss':loss_sum/n,'accuracy':correct/n,'samples':n},truth,pred

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-size',type=int,default=96)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--skip-test',action='store_true')
    args=p.parse_args();os.chdir(ROOT);torch.set_num_threads(args.threads);seed_all()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=ROOT/'results';out.mkdir(exist_ok=True);cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    environment={'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
                 'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'cpu_threads':args.threads,
                 'image_size':args.image_size,'epochs':args.epochs,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(out/'environment.json',environment)
    dsets,transform,split=loaders_data(args.image_size,args.workers);write_json(out/'split.json',split)
    caches={};extract_times={};all_results=[]
    for cfg0 in EXPERIMENTS:
        cfg=cfg0|{'epochs':args.epochs,'seed':42,'image_size':args.image_size,'optimizer':'SGD','momentum':.9,
                  'weight_decay':0.,'pretrained':'IMAGENET1K_V1','cache_features':cfg0['mode']=='head'}
        folder=out/cfg['name'];folder.mkdir(exist_ok=True)
        if (folder/'summary.json').exists():
            prior=json.loads((folder/'config.json').read_text())
            if prior!=cfg:raise RuntimeError('Existing run config differs; use a new results directory')
            all_results.append(json.loads((folder/'summary.json').read_text()));continue
        seed_all();model=make_model(cfg['model'],cfg['mode']).to(device)
        print('\nEXPERIMENT',cfg,flush=True)
        print(model,flush=True)
        (folder/'architecture.txt').write_text(str(model),encoding='utf-8')
        params={'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad]}
        write_json(folder/'parameters.json',params);write_json(folder/'config.json',cfg)
        if cfg['mode']=='head':
            key=cfg['model'];cp=cache/f'{key}_{args.image_size}_features.pt'
            if key not in caches:
                if cp.exists():
                    saved=torch.load(cp,weights_only=True,map_location='cpu')
                    caches[key]={k:TensorDataset(saved[k+'_x'],saved[k+'_y']) for k in ('train','val')}
                    extract_times[key]=saved['extraction_seconds']
                else:
                    extracted={};seconds=0.
                    for part in ['train','val']:
                        extracted[part],dt=extract(model,key,dsets[part],device,64,args.workers,part);seconds+=dt
                    caches[key]=extracted;extract_times[key]=seconds
                    torch.save({part+'_'+axis:extracted[part].tensors[j] for part in ['train','val'] for j,axis in enumerate(['x','y'])}
                               |{'extraction_seconds':seconds},cp)
            # Cache is a fixed, deterministic backbone evaluation: same activations each epoch.
            trainsets=caches[key];trainmodel=head(model,key)
        else:trainsets=dsets;trainmodel=model
        generator=torch.Generator().manual_seed(42)
        dls={part:DataLoader(trainsets[part],batch_size=cfg['batch_size'],shuffle=part=='train',
             generator=generator if part=='train' else None,num_workers=0 if cfg['mode']=='head' else args.workers) for part in ['train','val']}
        optimizer=torch.optim.SGD((p for p in trainmodel.parameters() if p.requires_grad),lr=cfg['lr'],momentum=.9)
        writer=SummaryWriter(str(folder/'tensorboard'));best=-1.;best_epoch=0;rows=[];start=time.perf_counter()
        try:
            for epoch in range(1,args.epochs+1):
                training_mode(model,cfg['model'],cfg['mode'])
                tr,_,_=run_epoch(trainmodel,dls['train'],device,optimizer,label=f"{cfg['name']} epoch {epoch} TRAIN")
                trainmodel.eval();va,_,_=run_epoch(trainmodel,dls['val'],device,label=f"{cfg['name']} epoch {epoch} VAL")
                row={'epoch':epoch,'train_loss':tr['loss'],'train_accuracy':tr['accuracy'],'val_loss':va['loss'],'val_accuracy':va['accuracy']};rows.append(row)
                for key,value in row.items():
                    if key!='epoch':writer.add_scalar(key,value,epoch)
                writer.flush()
                with (folder/'history.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
                if va['accuracy']>best:
                    best=va['accuracy'];best_epoch=epoch;torch.save({'model':model.state_dict(),'epoch':epoch},folder/'best.pt')
                print('EPOCH_RESULT',row,flush=True)
        finally:writer.close()
        summary={'name':cfg['name'],'best_val_accuracy':best,'best_epoch':best_epoch,'optimization_seconds':time.perf_counter()-start,
                 'feature_extraction_seconds':extract_times.get(cfg['model'],0.) if cfg['mode']=='head' else 0.,'trainable_parameters':params['trainable']}
        write_json(folder/'summary.json',summary);all_results.append(summary)
        del model,trainmodel,optimizer
    write_json(out/'validation_comparison.json',all_results)
    # Lock all candidates and select using validation BEFORE reading test labels.
    winner=max(all_results,key=lambda x:x['best_val_accuracy'])['name']
    chosen=list(dict.fromkeys(['resnet_baseline','densenet_baseline',winner]))
    write_json(out/'selection_before_test.json',{'criterion':'highest validation accuracy; first in predefined order breaks ties',
               'winner':winner,'evaluate_final_models':chosen,'candidates':all_results,'selected_at_unix':time.time()})
    if args.skip_test:return
    test=datasets.CIFAR10(str(ROOT/'data'),train=False,download=False,transform=transform)
    for name in chosen:
        folder=out/name
        if (folder/'test_metrics.json').exists():continue
        cfg=json.loads((folder/'config.json').read_text());model=make_model(cfg['model'],cfg['mode'],pretrained=False).to(device)
        state=torch.load(folder/'best.pt',map_location=device,weights_only=True);model.load_state_dict(state['model']);model.eval()
        metrics,truth,pred=run_epoch(model,DataLoader(test,batch_size=64,num_workers=args.workers),device,label=name+' TEST',predict=True)
        metrics|={'checkpoint_epoch':state['epoch'],'classification_report':classification_report(truth,pred,target_names=CLASSES,output_dict=True,zero_division=0),
                  'confusion_matrix':confusion_matrix(truth,pred,labels=list(range(10))).tolist()}
        write_json(folder/'test_metrics.json',metrics)
        with (folder/'test_predictions.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['test_index','target','prediction']);w.writerows((i,y,p) for i,(y,p) in enumerate(zip(truth,pred)))
        print('TEST_RESULT',name,metrics['accuracy'],flush=True)
    print('ALL EXPERIMENTS COMPLETE. Selected:',winner,flush=True)


if __name__=='__main__':main()
