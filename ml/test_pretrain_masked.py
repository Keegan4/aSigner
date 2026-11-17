"""Checks masked-landmark pretraining on made-up clips. Says nothing about whether it helps real accuracy."""
import json

import numpy as np
import torch

from pretrain import pretrain
from pretrain_masked import SOURCE_NOTE, Filler, average_guess, hidden_error, hide, pretrain_masked
from test_pretrain import write_casl, write_samples
from test_train import fake_clip
from train import SignNet, model_inputs

from data import Clip
from app.recognition.features import CLIP_FRAMES, COORDINATE_SIZE, NORMALISED_PARTS, NORMALISED_SIZE, REGION_PARTS


def inputs(count=6):
    rng = np.random.default_rng(0)
    return torch.from_numpy(model_inputs([Clip(fake_clip(rng, i % 3), "A", "u1") for i in range(count)]))


def test_hidden_parts_look_lost_and_only_tracked_ones_are_asked_for():
    x = inputs()
    shown, wanted = hide(x, torch.Generator().manual_seed(1))
    assert shown.shape == x.shape and wanted.shape == (len(x), CLIP_FRAMES, COORDINATE_SIZE)
    asked = wanted.bool()
    assert asked.any() and not asked.all()
    # What is asked for was tracked, and is now zero with its part marked absent.
    assert (x[..., :COORDINATE_SIZE][asked] != 0).any() and not shown[..., :COORDINATE_SIZE][asked].any()
    right_hand = REGION_PARTS[1][0]
    gone = wanted[..., NORMALISED_PARTS[right_hand]].any(dim=-1)
    assert not shown[..., COORDINATE_SIZE + right_hand][gone].any()
    # Everything not asked for is passed through untouched where it was tracked at all.
    kept = ~asked & (shown[..., :COORDINATE_SIZE] != 0)
    assert torch.equal(shown[..., :COORDINATE_SIZE][kept], x[..., :COORDINATE_SIZE][kept])
    # The fake clips have no left hand: nothing to predict there, however much of it is hidden.
    assert not wanted[..., NORMALISED_PARTS[REGION_PARTS[0][0]]].any()
    # A part is hidden for a stretch of the clip, never all of it.
    share = gone.float().mean().item()
    assert 0.15 < share < 0.6


def test_error_counts_hidden_coordinates_only_and_the_average_guess_is_exact_for_a_still_hand():
    x = inputs()
    shown, wanted = hide(x, torch.Generator().manual_seed(2))
    perfect = x[..., :COORDINATE_SIZE].clone()
    assert hidden_error(perfect, x, wanted) == 0
    perfect[~wanted.bool()] += 5.0  # wrong only where nothing was asked
    assert hidden_error(perfect, x, wanted) == 0
    assert hidden_error(torch.zeros_like(perfect), x, wanted) > 0

    still = x[:, :1].expand(-1, CLIP_FRAMES, -1).clone()
    shown, wanted = hide(still, torch.Generator().manual_seed(3))
    assert hidden_error(average_guess(shown), still, wanted) < 1e-6


def test_filler_names_every_frame_and_the_model_still_classifies_from_its_summary_token():
    net = SignNet(4).eval()
    x = inputs(3)
    assert net.encode(x).shape == (3, CLIP_FRAMES + 1, 320)
    assert torch.equal(net(x), net.head(net.encode(x)[:, 0]))
    assert Filler(net, 320)(x).shape == (3, CLIP_FRAMES, COORDINATE_SIZE)
    assert x.shape[-1] == NORMALISED_SIZE


def test_masked_pretraining_learns_and_pretrain_can_start_from_it(tmp_path):
    write_samples(tmp_path / "public", per_letter=16)
    write_casl(tmp_path / "casl")
    report = pretrain_masked(tmp_path / "public", tmp_path / "masked.pt", epochs=6, seed=0, casl=tmp_path / "casl")
    history = report["history"]
    assert [row["epoch"] for row in history] == list(range(1, 7))
    assert set(history[0]) == {"epoch", "train", "asl-now", "casl-w60"}
    assert history[-1]["train"]["loss"] < history[0]["train"]["loss"]
    assert history[-1]["asl-now"]["average_guess"] == history[0]["asl-now"]["average_guess"]  # same parts hidden
    assert report["source"].startswith(SOURCE_NOTE)
    assert json.loads((tmp_path / "masked.json").read_text())["history"] == history

    saved = torch.load(tmp_path / "masked.pt", weights_only=True)
    assert saved["labels"] == [] and not any(key.startswith("decoder") for key in saved["state"])

    started = pretrain(tmp_path / "public", tmp_path / "p.pt", epochs=1, seed=0, casl=tmp_path / "casl",
                       init=tmp_path / "masked.pt")
    assert "started from masked.pt" in started["source"] and SOURCE_NOTE in started["source"]
