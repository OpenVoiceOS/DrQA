import torch

from drqa.utils import BatchGen


def _row(idx, n_tokens):
    context_ids = list(range(2, n_tokens + 2))
    features = [(True, False, True, 0.5)] * n_tokens
    tags = [0] * n_tokens
    ents = [0] * n_tokens
    question_ids = [1, 2, 3]
    text = "text"
    span = [(i, i + 1) for i in range(n_tokens)]
    return (idx, context_ids, features, tags, ents, question_ids, text, span)


def test_batchgen_eval_pads_and_masks():
    BatchGen.pos_size = 2
    BatchGen.ner_size = 2
    data = [_row(0, 3), _row(1, 5)]
    batches = list(BatchGen(data, batch_size=2, gpu=False, evaluation=True))
    assert len(batches) == 1
    batch = batches[0]
    context_id, context_feature, context_tag, _context_ent, context_mask, \
        question_id, _question_mask, text, _span = batch

    assert context_id.shape == (2, 5)
    assert context_feature.shape == (2, 5, 4)
    assert context_tag.shape == (2, 5, 2)
    assert question_id.shape == (2, 3)
    # the padded sequence (length 3) has two masked positions
    assert context_mask.sum().item() == 2
    assert text == ["text", "text"]


def test_batchgen_train_yields_lengths():
    BatchGen.pos_size = 2
    BatchGen.ner_size = 2
    row = _row(0, 4) + (0, 3)
    batches = list(BatchGen([row], batch_size=1, gpu=False, evaluation=False))
    assert len(batches) == 1
    assert len(batches[0]) == 11  # incl. y_s and y_e
    y_s, y_e = batches[0][7], batches[0][8]
    assert isinstance(y_s, torch.Tensor)
    assert y_s.item() == 0 and y_e.item() == 3
