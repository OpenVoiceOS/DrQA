import msgpack
import torch

from drqa.model import DocReaderModel
from drqa.utils import annotate, init, to_id, BatchGen


class DrQA:
    def __init__(self, model_path, meta_path, cuda=torch.cuda.is_available()):
        if cuda:
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.cuda = cuda
        state_dict = checkpoint['state_dict']
        opt = checkpoint['config']
        with open(meta_path, 'rb') as f:
            self.meta = msgpack.load(f, encoding='utf8')

        embedding = torch.Tensor(self.meta['embedding'])
        opt['pretrained_words'] = True
        opt['vocab_size'] = embedding.size(0)
        opt['embedding_dim'] = embedding.size(1)
        opt['pos_size'] = len(self.meta['vocab_tag'])
        opt['ner_size'] = len(self.meta['vocab_ent'])
        opt['cuda'] = cuda
        BatchGen.pos_size = opt['pos_size']
        BatchGen.ner_size = opt['ner_size']
        self.model = DocReaderModel(opt, embedding, state_dict)
        self.w2id = {w: i for i, w in enumerate(self.meta['vocab'])}
        self.tag2id = {w: i for i, w in enumerate(self.meta['vocab_tag'])}
        self.ent2id = {w: i for i, w in enumerate(self.meta['vocab_ent'])}
        init()  # spacy
        self._id = 0

    def predict(self, evidence, question):
        self._id += 1
        annotated = annotate((f'interact-{self._id}', evidence, question), self.meta['wv_cased'])
        model_in = to_id(annotated, self.w2id, self.tag2id, self.ent2id)
        model_in = next(iter(BatchGen([model_in], batch_size=1, gpu=self.cuda, evaluation=True)))
        prediction = self.model.predict(model_in)[0]
        return prediction


