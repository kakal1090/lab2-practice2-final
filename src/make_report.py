"""Build tables, charts and Vietnamese report exclusively from saved real results."""
import base64,csv,html,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .experiment import ROOT,EXPERIMENTS,CLASSES


def main():
    results=ROOT/'results';report=ROOT/'report';fig=report/'figures';fig.mkdir(parents=True,exist_ok=True)
    env=json.loads((results/'environment.json').read_text());selection=json.loads((results/'selection_before_test.json').read_text())
    runs=[]
    for c in EXPERIMENTS:
        folder=results/c['name'];summary=json.loads((folder/'summary.json').read_text());params=json.loads((folder/'parameters.json').read_text())
        with (folder/'history.csv').open() as f:history=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        test=json.loads((folder/'test_metrics.json').read_text()) if (folder/'test_metrics.json').exists() else None
        best=next(x for x in history if x['epoch']==summary['best_epoch'])
        runs.append({'cfg':c,'summary':summary,'parameters':params,'history':history,'test':test,'best':best})
    plt.rcParams.update({'font.size':10,'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#2458a6','#d47717','#138565','#9a4b9f','#b54444']
    for metric in ['train_loss','val_loss','train_accuracy','val_accuracy']:
        f,ax=plt.subplots(figsize=(9,4.7))
        for i,r in enumerate(runs):
            ax.plot([x['epoch'] for x in r['history']],[x[metric] for x in r['history']],marker='o',label=r['cfg']['name'],color=colors[i])
        ax.set_xlabel('Epoch');ax.set_ylabel(metric);ax.set_xticks(range(1,env['epochs']+1));ax.grid(alpha=.2)
        if 'accuracy' in metric:ax.set_ylim(0,1)
        else:ax.set_ylim(bottom=0)
        ax.legend(fontsize=8);f.tight_layout();f.savefig(fig/f'{metric}.png');plt.close(f)
    f,ax=plt.subplots(figsize=(9,4.7))
    names=[r['cfg']['name'] for r in runs];values=[r['summary']['best_val_accuracy']*100 for r in runs]
    bars=ax.barh(names,values,color=colors);ax.set_xlim(0,100);ax.set_xlabel('Best validation accuracy (%)')
    for b,v in zip(bars,values):ax.text(v+.5,b.get_y()+b.get_height()/2,f'{v:.2f}%',va='center')
    ax.grid(axis='x',alpha=.2);f.tight_layout();f.savefig(fig/'comparison.png');plt.close(f)
    for r in runs:
        if r['test']:
            a=np.array(r['test']['confusion_matrix']);f,ax=plt.subplots(figsize=(8,7));im=ax.imshow(a,cmap='Blues')
            ax.set_xticks(range(10),CLASSES,rotation=45,ha='right');ax.set_yticks(range(10),CLASSES)
            ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(r['cfg']['name']+' - test confusion matrix')
            for i in range(10):
                for j in range(10):ax.text(j,i,str(a[i,j]),ha='center',va='center',fontsize=8,color='white' if a[i,j]>a.max()/2 else 'black')
            f.colorbar(im,ax=ax,shrink=.8);f.tight_layout();f.savefig(fig/(r['cfg']['name']+'_confusion.png'));plt.close(f)
    rows=[]
    for r in runs:
        c,s,b=r['cfg'],r['summary'],r['best']
        rows.append([c['name'],c['model'],c['mode'],c['lr'],c['batch_size'],s['trainable_parameters'],s['best_epoch'],
                     round(b['train_accuracy']*100,4),round(s['best_val_accuracy']*100,4),round(b['val_loss'],6),
                     round(r['test']['accuracy']*100,4) if r['test'] else '',round(s['optimization_seconds'],2),round(s['feature_extraction_seconds'],2)])
    headers=['Run','Model','Mode','LR','Batch','Trainable','Best epoch','Train acc (%)','Val acc (%)','Val loss','Test acc (%)','Optimization seconds','Feature extraction seconds']
    with (results/'comparison.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(headers);w.writerows(rows)
    compact_headers=['Run','LR','Batch','Best epoch','Val acc (%)','Test acc (%)']
    compact_rows=[[row[i] for i in [0,3,4,6,8,10]] for row in rows]
    table='| '+' | '.join(compact_headers)+' |\n| '+' | '.join(['---']*len(compact_headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in compact_rows)
    time_table='| Run | Optimization (s) | Feature extraction dùng chung (s) |\n|---|---:|---:|\n'+''.join(f'| {row[0]} | {row[11]} | {row[12]} |\n' for row in rows)

    base=runs[0];dense=runs[1];lr=runs[2];batch=runs[3];fine=runs[4];winner=next(r for r in runs if r['cfg']['name']==selection['winner'])
    def pp(r):return (r['summary']['best_val_accuracy']-base['summary']['best_val_accuracy'])*100
    lines=[
    '# PRACTICE 2 - HUẤN LUYỆN VÀ THÍCH NGHI MẠNG PRETRAINED',
    'Thông tin nhóm: [Điền họ tên và MSSV của ba thành viên] · Lớp: [Điền] · Giảng viên: [Điền].',
    'Bài thực nghiệm trên CIFAR-10. Các bảng được sinh từ log chạy thật; không dùng số liệu mô phỏng.',
    f'Tóm tắt: Thực hiện năm lượt huấn luyện trên đầy đủ CIFAR-10 với ResNet18 và DenseNet121 pretrained ImageNet, so sánh kiến trúc, learning rate, batch size và fine-tuning. Giữ cùng split, preprocessing 96×96, seed 42 và 5 epoch. Cấu hình được chọn bằng validation là {selection["winner"]}, đạt validation {winner["summary"]["best_val_accuracy"]:.2%}'+(f' và test {winner["test"]["accuracy"]:.2%} trên 10.000 ảnh.' if winner['test'] else '.'),
    '## 1. Đối chiếu yêu cầu đề',
    '| Yêu cầu slide 23 | Bằng chứng trong bài |\n|---|---|\n| Cài PyTorch/torchvision | requirements.txt, results/environment.json |\n| Nạp pretrained | ResNet18 và DenseNet121, IMAGENET1K_V1 |\n| Xem kiến trúc và tham số | results/<run>/architecture.txt, parameters.json; notebook in mô hình |\n| Thay lớp cuối và freeze/fine-tune | fc/classifier 10 lớp; head hoặc layer4+fc |\n| transforms và DataLoader | src/experiment.py: loaders_data, DataLoader ảnh và feature batches |\n| Loss/optimizer/training loop | CrossEntropyLoss, SGD momentum 0.9, run_epoch |\n| Đánh giá test | test_metrics.json, confusion matrix, classification report |\n| Thử nhiều kiến trúc | resnet_baseline / densenet_baseline |\n| Đổi hyperparameters | resnet_lr_01 / resnet_batch64 |\n| TensorBoard | results/<run>/tensorboard, bốn scalar theo epoch |',
    '## 2. Dữ liệu và thiết kế thực nghiệm',
    f'CIFAR-10 có 50.000 ảnh train gốc và 10.000 ảnh test, mỗi ảnh RGB 32×32, 10 lớp. Từ train gốc, tách phân tầng 45.000 train và 5.000 validation (500/lớp) với seed 42. Dùng đầy đủ dữ liệu, không dùng tập con. Chỉ số split lưu trong results/split.json. Test tách biệt hoàn toàn, không chọn cấu hình bằng test.',
    f'Ảnh được resize {env["image_size"]}×{env["image_size"]}, ToTensor và normalize mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225). Không augmentation. Kích thước 96 là lựa chọn thực nghiệm phù hợp CPU; đề không bắt buộc 224. Cùng preprocessing cho cả năm lượt. Không trộn kết quả với thí nghiệm cũ 224×224 trong repo nhóm.',
    f'Mỗi lượt {env["epochs"]} epoch, SGD momentum 0.9, weight_decay=0, CrossEntropyLoss trên logits (không thêm softmax trước loss). Các lượt khởi tạo từ pretrained ImageNet và cùng khởi tạo head theo seed 42; fine-tuning không lấy checkpoint baseline để tránh thêm ngân sách học. Chọn best checkpoint theo validation accuracy; nếu hòa giữ epoch xuất hiện đầu tiên.',
    '## 3. Kiến trúc, transfer learning và fine-tuning',
    f'ResNet18 dùng residual connections. Thay fc 512→1000 bằng Linear(512,10); mode head chỉ học fc.weight/fc.bias, tổng {base["parameters"]["trainable"]:,} tham số trên {base["parameters"]["total"]:,} tham số mô hình. DenseNet121 nối đặc trưng qua dense connections; classifier mới Linear(1024,10), học {dense["parameters"]["trainable"]:,}/{dense["parameters"]["total"]:,} tham số.',
    f'Fine-tuning ResNet18 mở layer4 và fc, tổng {fine["parameters"]["trainable"]:,} tham số. Các lớp trước frozen và eval; layer4 và fc ở train. Với mode head, toàn bộ backbone ở eval để không đổi BatchNorm running statistics.',
    'Các lượt head dùng cache đặc trưng để tiết kiệm CPU. Backbone frozen, eval và preprocessing xác định nên cùng ảnh luôn có cùng feature; học head trên cache tương đương học head trên output backbone cố định, trong sai số tính toán. Đã kiểm tra forward cached và full model khớp. Không cache backbone cho fine-tuning vì trọng số thay đổi. Đây là tối ưu tính toán, không bổ sung dữ liệu hay thay đổi bài toán.',
    '## 4. Kết quả validation và test',table,time_table,
    'Các ô test trống là cấu hình không được đánh giá test theo kế hoạch, không phải accuracy bằng 0. Sau khi kết thúc toàn bộ lượt validation, khóa lựa chọn trong selection_before_test.json. Test hai kiến trúc baseline và cấu hình thắng validation (nếu khác baseline) trên đủ 10.000 ảnh, không điều chỉnh lại hyperparameter sau khi xem test.',
    'Thời gian optimization của lượt head KHÔNG gồm trích xuất đặc trưng; cột feature extraction là chi phí dùng chung theo kiến trúc. Không dùng riêng optimization seconds để tuyên bố model này nhanh hơn model khác. Cùng chi phí extraction được ghi lại ở các lượt ResNet head và không cộng lặp khi tính ngân sách toàn bộ.',
    '## 5. Phân tích kết quả',
    f'- So sánh kiến trúc: DenseNet121 đạt validation {dense["summary"]["best_val_accuracy"]:.2%}, so với ResNet18 {base["summary"]["best_val_accuracy"]:.2%}, chênh lệch {pp(dense):+.2f} điểm phần trăm trong điều kiện này. Chênh lệch có thể do kiến trúc và đặc trưng pretrained, không đủ để kết luận một kiến trúc luôn tốt hơn.',
    f'- Learning rate: 0.01 đạt {lr["summary"]["best_val_accuracy"]:.2%}, thay đổi {pp(lr):+.2f} điểm phần trăm so với 0.001. LR điều khiển độ lớn bước cập nhật; đánh giá cùng đường loss để xem hội tụ/dao động, không chỉ accuracy cuối.',
    f'- Batch size: 64 đạt {batch["summary"]["best_val_accuracy"]:.2%}, thay đổi {pp(batch):+.2f} điểm phần trăm so với 32. Với 45.000 ảnh, batch 32 có 1.407 bước/epoch, batch 64 có 704 bước; ngân sách ảnh/epoch bằng nhau nhưng số optimizer steps khác nhau. Đây là giới hạn quan trọng khi diễn giải.',
    f'- Fine-tuning: layer4+fc đạt {fine["summary"]["best_val_accuracy"]:.2%}, thay đổi {pp(fine):+.2f} điểm phần trăm so với chỉ học head. Mở thêm lớp tăng khả năng thích nghi và chi phí tính toán; có thể tăng nguy cơ overfitting. Không chọn thêm LR sau khi xem test.',
    f'- Cấu hình được chọn bằng validation: {selection["winner"]}, epoch {winner["summary"]["best_epoch"]}, validation {winner["summary"]["best_val_accuracy"]:.2%}.',
    ]
    # Describe concrete changes in the saved curves rather than inventing causes.
    prior=lr['history'][2];unstable=lr['history'][3]
    lines.append(f'Ở lượt LR=0.01, từ epoch 3 sang 4, validation loss {prior["val_loss"]:.4f} → {unstable["val_loss"]:.4f}, accuracy {prior["val_accuracy"]:.2%} → {unstable["val_accuracy"]:.2%}. Dao động này cùng mức best thấp hơn baseline cho thấy LR=0.01 chưa phù hợp trong cấu hình đang xét. Không thể suy ra mọi LR lớn đều kém từ một thử nghiệm.')
    db=dense['best'];dl=dense['history'][-1]
    lines.append(f'DenseNet đạt best ở epoch {int(db["epoch"])}. Từ đó đến epoch cuối, train loss {db["train_loss"]:.4f} → {dl["train_loss"]:.4f}, trong khi validation loss {db["val_loss"]:.4f} → {dl["val_loss"]:.4f}. Mẫu này gợi ý lợi ích tổng quát hóa đã chững lại; lưu best validation checkpoint giúp tránh mặc định dùng epoch cuối.')
    low=min(fine['history'],key=lambda r:r['val_loss'])
    if int(low['epoch'])!=fine['summary']['best_epoch']:
        lines.append(f'Ở fine-tuning, validation loss thấp nhất tại epoch {int(low["epoch"])} ({low["val_loss"]:.4f}), còn validation accuracy cao nhất tại epoch {fine["summary"]["best_epoch"]}. Hai thước đo có thể khác nhau: accuracy chỉ xét nhãn argmax, CrossEntropy còn xét xác suất gán cho nhãn đúng. Vì vậy loss có thể tăng khi một số dự đoán sai trở nên quá tự tin dù số dự đoán đúng tăng. Bài giữ tiêu chí accuracy đã chọn từ đầu, đồng thời ghi nhận rủi ro overfitting; không đổi tiêu chí sau khi thấy test.')
    for r in runs:
        first,last=r['history'][0],r['history'][-1]
        gap=(last['train_accuracy']-last['val_accuracy'])*100
        lines.append(f'- {r["cfg"]["name"]}: train loss {first["train_loss"]:.4f} → {last["train_loss"]:.4f}; val loss {first["val_loss"]:.4f} → {last["val_loss"]:.4f}; khoảng cách train−val accuracy cuối {gap:+.2f} điểm %.')
    lines.append('Train accuracy được đo ngay trong vòng lặp cập nhật trọng số; validation được đo sau epoch. Vì vậy train có thể thấp hơn validation. Với fine-tuning còn có khác biệt chế độ BatchNorm train/eval ở layer4. Các khoảng cách trên mô tả đường học, không tự động chứng minh hoặc loại trừ overfitting chỉ từ một epoch.')
    if winner['test']:
        t=winner['test'];lines.append(f'Cấu hình được chọn đạt test accuracy **{t["accuracy"]:.2%}**, macro F1 **{t["classification_report"]["macro avg"]["f1-score"]:.4f}**. Chênh validation−test {(winner["summary"]["best_val_accuracy"]-t["accuracy"])*100:+.2f} điểm %.')
        matrix=np.array(t['confusion_matrix']);np.fill_diagonal(matrix,0);a,b=np.unravel_index(matrix.argmax(),matrix.shape)
        lines.append(f'Cặp nhầm có số lượng lớn nhất theo một chiều trong test: ảnh thật {CLASSES[a]} bị dự đoán {CLASSES[b]} ({matrix[a,b]} ảnh). Đây là quan sát từ confusion matrix; cần xem ảnh cụ thể trước khi quy nguyên nhân.')
        lines.append('### Precision / Recall / F1 của cấu hình được chọn')
        class_table='| Lớp | Precision | Recall | F1 | Số ảnh |\n|---|---:|---:|---:|---:|\n'
        for c in CLASSES:
            v=t['classification_report'][c];class_table+=f'| {c} | {v["precision"]:.4f} | {v["recall"]:.4f} | {v["f1-score"]:.4f} | {int(v["support"])} |\n'
        lines.append(class_table)
    lines += ['## 6. Đồ thị và TensorBoard']
    for name in ['comparison','train_loss','val_loss','train_accuracy','val_accuracy']:
        lines.append(f'![{name}](figures/{name}.png)')
    for r in runs:
        if r['test']:lines.append(f'![{r["cfg"]["name"]} confusion matrix](figures/{r["cfg"]["name"]}_confusion.png)')
    lines += ['Các PNG được vẽ từ history.csv chưa smoothing. Bằng chứng dùng TensorBoard là event files trong results/<run>/tensorboard; mở bằng `python -m tensorboard.main --logdir results`. Cả bốn scalar được ghi theo epoch, có thể so sánh nhiều run. PNG không thay thế event logs.',
    '## 7. Hạn chế và hướng phát triển',
    'Một seed và 5 epoch chưa đủ đánh giá độ ổn định thống kê. Resize 96 không phải recipe ImageNet chuẩn 224; kết quả chỉ áp dụng cho pipeline này. Không augmentation hoặc lịch LR để giữ so sánh đơn giản, nên đây không phải kết quả tối ưu CIFAR-10. Việc chọn trong nhiều cấu hình có thể làm validation lạc quan; giữ test làm đánh giá cuối. Hướng phát triển: chạy thêm seed, augmentation và LR schedule với kế hoạch chọn bằng validation trước khi dùng test mới.',
    '## 8. Cách chạy lại',
    'Python >=3.11. Cài requirements. Từ thư mục gốc chạy `python -m src.download_data`, rồi `python -u -m src.experiment --image-size 96 --epochs 5`, sau đó `python -m src.make_report`. Chạy kiểm tra bằng `python -m unittest discover -s tests -v`. Có notebook Colab trọn bài và cách upload archive khi nguồn mạng chậm. Muốn chạy cấu hình khác phải dùng thư mục kết quả mới, không trộn log.',
    'Môi trường thực tế:\n\n```json\n'+json.dumps(env,ensure_ascii=False,indent=2)+'\n```',
    '## 9. Tài liệu tham khảo',
    '- Đề lớp: Practice 2, Chapter 3, slide 23; ví dụ CIFAR-10 ở slide 24. PDF đề không đưa vào gói để công khai lại.\n- Alex Krizhevsky (2009), Learning Multiple Layers of Features from Tiny Images; https://cave.cs.toronto.edu/kriz/cifar.html\n- PyTorch, Transfer Learning for Computer Vision Tutorial; https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial\n- Torchvision, Models and pre-trained weights; https://docs.pytorch.org/vision/stable/models.html',
    '## 10. Gợi ý trả lời khi trình bày',
    '- Vì sao thay lớp cuối? ImageNet có 1.000 lớp, CIFAR-10 có 10 lớp.\n- Vì sao freeze? Tận dụng feature đã học và giảm tham số cần cập nhật.\n- Fine-tuning khác gì? Mở layer4 để feature thích nghi với dữ liệu đích.\n- Vì sao không dùng test chọn LR? Sẽ làm rò rỉ thông tin đánh giá vào lựa chọn cấu hình.\n- Vì sao val accuracy có thể cao hơn train? Train được đo trong lúc trọng số thay đổi, validation sau epoch; đây chưa phải bằng chứng sai dữ liệu.\n- Vì sao thời gian head nhỏ? Đã tách chi phí trích xuất đặc trưng; phải tính cả chi phí đó khi so sánh.\n- Vì sao không kết luận “luôn tốt hơn”? Một dataset, một seed và ngân sách nhỏ không đủ khái quát.'
    ]
    text='\n\n'.join(lines).replace('results/<run>/','results/[run]/');(report/'BAO_CAO_PRACTICE_2.md').write_text(text,encoding='utf-8')
    # A portable HTML version with all figures embedded; markdown is rendered when available.
    import markdown
    content=markdown.markdown(text,extensions=['tables','fenced_code'])
    for image in fig.glob('*.png'):
        content=content.replace('src="figures/'+image.name+'"','src="data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()+'"')
    document='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Practice 2 - Báo cáo thực nghiệm</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#23334a;max-width:1150px;margin:48px auto;padding:0 30px}h1{font-size:32px;color:#183b6b}h2{margin-top:38px;color:#183b6b;border-bottom:2px solid #d9e4f1;padding-bottom:8px}table{border-collapse:collapse;width:100%;font-size:12px;display:block;overflow:auto}th,td{border:1px solid #d6dfe9;padding:8px;text-align:left}th{background:#e9eff7}tr:nth-child(even){background:#f7f9fc}img{max-width:100%;display:block;margin:20px auto}pre{background:#eef2f7;padding:16px;overflow:auto;font-size:12px}code{font-size:13px}@media print{body{margin:0;font-size:11px}h2{break-after:avoid}img,table{break-inside:avoid}}
</style><body>'''+content+'</body></html>'
    (report/'BAO_CAO_PRACTICE_2.html').write_text(document,encoding='utf-8')
    print('REPORT GENERATED',flush=True)

if __name__=='__main__':main()
