import unittest
import torch
from src.experiment import make_model,head,features,training_mode,split_indices

class PracticeChecks(unittest.TestCase):
    def test_stratified_split(self):
        targets=[c for c in range(10) for _ in range(5000)]
        tr,va=split_indices(targets)
        self.assertEqual((len(set(tr)),len(set(va))),(45000,5000))
        self.assertFalse(set(tr)&set(va))
        self.assertEqual((tr,va),split_indices(targets))
        for c in range(10):self.assertEqual(sum(targets[i]==c for i in va),500)

    def test_cached_features_match_forward_and_freeze(self):
        torch.set_num_threads(2)
        for name in ['resnet18','densenet121']:
            m=make_model(name,pretrained=False).eval();x=torch.randn(2,3,96,96)
            with torch.no_grad():
                full=m(x);z=features(m,name,x);cached=head(m,name)(z)
            torch.testing.assert_close(full,cached,rtol=1e-5,atol=1e-6)
            before={n:v.clone() for n,v in m.state_dict().items() if not n.startswith(('fc.','classifier.'))}
            training_mode(m,name,'head');opt=torch.optim.SGD(head(m,name).parameters(),lr=.01)
            opt.zero_grad();torch.nn.functional.cross_entropy(head(m,name)(z),torch.tensor([1,2])).backward();opt.step()
            for n,v in before.items():torch.testing.assert_close(v,m.state_dict()[n],rtol=0,atol=0)

    def test_finetune_scope(self):
        m=make_model('resnet18','finetune',False);training_mode(m,'resnet18','finetune')
        self.assertTrue(m.layer4.training);self.assertFalse(m.layer3.training)
        names=[n for n,p in m.named_parameters() if p.requires_grad]
        self.assertTrue(all(n.startswith(('layer4.','fc.')) for n in names))
        self.assertTrue(any(n.startswith('layer4.') for n in names))

if __name__=='__main__':unittest.main()
