"""Official CIFAR-10 archive, resumable HTTP ranges, final MD5 verification."""
import concurrent.futures
import hashlib
import math
from pathlib import Path
from urllib.request import Request, urlopen
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import extract_archive


def md5(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'md5').hexdigest()


def prepare(root='data', workers=16):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    archive=root/CIFAR10.filename
    if archive.exists() and md5(archive)==CIFAR10.tgz_md5:
        print('Verified existing archive',flush=True)
    else:
        url='https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
        with urlopen(Request(url,headers={'Range':'bytes=0-0'}),timeout=30) as r:
            if r.status!=206:raise RuntimeError('Official server does not support Range; upload verified archive instead.')
            total=int(r.headers['Content-Range'].split('/')[-1]);r.read()
        chunk=math.ceil(total/workers);parts=root/'download_parts';parts.mkdir(exist_ok=True)
        def fetch(i):
            start=i*chunk;end=min(total-1,(i+1)*chunk-1);p=parts/f'{i:03}.part'
            have=p.stat().st_size if p.exists() else 0
            if have>end-start+1:raise RuntimeError('Oversized partial file')
            if have<end-start+1:
                first=start+have
                with urlopen(Request(url,headers={'Range':f'bytes={first}-{end}'}),timeout=60) as r:
                    if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {first}-{end}/'):
                        raise RuntimeError('Unexpected range response; refusing append')
                    with p.open('ab') as f:
                        while block:=r.read1(256*1024):f.write(block)
            if p.stat().st_size!=end-start+1:raise RuntimeError('Incomplete part; rerun to resume')
            print(f'Download part {i+1}/{workers} complete',flush=True)
            return p
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers,8)) as pool:
            files=list(pool.map(fetch,range(workers)))
        assembled=root/'cifar-10.assembled'
        with assembled.open('wb') as f:
            for part in files:
                with part.open('rb') as src:
                    while block:=src.read(1024*1024):f.write(block)
        if md5(assembled)!=CIFAR10.tgz_md5:raise RuntimeError('MD5 mismatch; do not use archive')
        assembled.replace(archive)
    extract_archive(str(archive),str(root))
    print('DATA READY',flush=True)


if __name__=='__main__':prepare()
