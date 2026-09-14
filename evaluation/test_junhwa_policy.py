import pytest
import torch
from torch import nn
from evaluation import junhwa_policy


def checkpoint(recurrent=True):
    torch.manual_seed(17)
    body = nn.Sequential(nn.Linear(214, 8), nn.Tanh())
    head = nn.Linear(8, 84)
    gru = nn.GRUCell(8, 8)
    state = {'actor_body.' + k: v for k, v in body.state_dict().items()}
    state.update({'actor_logits.' + k: v for k, v in head.state_dict().items()})
    if recurrent:
        state.update({'actor_gru.' + k: v for k, v in gru.state_dict().items()})
    return dict(model=state, cfg=dict(activation='tanh', num_bins=21,
                gru_last=recurrent, gru_all=False, continuous_action=False),
                norm=dict(mean=torch.zeros(214), var=torch.ones(214))), body, gru, head


@pytest.mark.parametrize('recurrent', [True, False])
def test_actor_matches_reference_and_resets_only_selected_lane(tmp_path, recurrent):
    ckpt, body, gru, head = checkpoint(recurrent)
    path = tmp_path / 'model.pt'; torch.save(ckpt, path)
    policy, norm = junhwa_policy.load_junhwa_policy(path, 'cpu')
    assert norm is None  # normalization is inside adapter, never twice
    state = policy.initial_state(3, 'cpu')
    reference = torch.zeros(3, 8)
    for starts in (torch.ones(3), torch.zeros(3), torch.tensor([0., 1., 0.])):
        obs = torch.randn(3, 214)
        x = body((obs / torch.sqrt(torch.tensor(1. + 1e-8))).clamp(-10, 10))
        if recurrent:
            reference = gru(x, reference * (1 - starts[:, None]))
            x = reference
        actions, state = policy.act(obs, state, starts)
        assert torch.equal(actions, head(x).reshape(3, 4, 21).argmax(-1))
        if recurrent:
            torch.testing.assert_close(state, reference)
    with pytest.raises(ValueError):
        policy.act(obs, state, starts, sample=True)


def test_rejects_unknown_actor_weights_and_invalid_norm(tmp_path):
    ckpt, *_ = checkpoint()
    ckpt['model']['actor_unknown.weight'] = torch.ones(1)
    path = tmp_path / 'bad.pt'; torch.save(ckpt, path)
    with pytest.raises(ValueError, match='actor'):
        junhwa_policy.load_junhwa_policy(path, 'cpu')
    del ckpt['model']['actor_unknown.weight']
    ckpt['norm']['var'][0] = -1
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match='normalization'):
        junhwa_policy.load_junhwa_policy(path, 'cpu')
