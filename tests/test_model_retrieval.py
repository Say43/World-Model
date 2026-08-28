"""Tests for src/model/retrieval.py frustum-overlap spatial-memory retrieval."""
import torch

from src.model.retrieval import frustum_overlap_scores, select_top_k


def _pose_at(x, y, z, yaw=0.0):
    """c2w pose at world position (x,y,z), facing -Y rotated by yaw around Z
    (arbitrary convention for this test; only relative geometry matters)."""
    c, s = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    pose = torch.eye(4)
    # +X right, +Y up (world Z-up convention means camera "up" here is
    # just picked as world +Z for a camera looking along the horizontal
    # plane), -Z forward -> forward direction rotates with yaw in the XY
    # plane.
    right = torch.tensor([c, s, 0.0])
    up = torch.tensor([0.0, 0.0, 1.0])
    forward = torch.tensor([-s, c, 0.0])  # look direction
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = torch.tensor([x, y, z])
    return pose


def test_overlap_score_high_for_same_pose():
    pose = _pose_at(0.0, 0.0, 1.5).unsqueeze(0)  # (1,4,4)
    intr = torch.tensor([[100.0, 100.0, 128.0, 128.0]])
    candidates = pose.unsqueeze(1)  # (1,1,4,4), identical to query
    cand_intr = intr.unsqueeze(1)
    scores = frustum_overlap_scores(pose, intr, candidates, cand_intr, width=256, height=256, distance_scale=5.0)
    assert scores.shape == (1, 1)
    assert scores[0, 0] > 0.9


def test_overlap_score_low_for_far_away_and_facing_away():
    query = _pose_at(0.0, 0.0, 1.5).unsqueeze(0)
    far = _pose_at(0.0, 100.0, 1.5, yaw=torch.pi).unsqueeze(0)  # far away, facing opposite
    intr = torch.tensor([[100.0, 100.0, 128.0, 128.0]])
    scores = frustum_overlap_scores(
        query, intr, far.unsqueeze(1), intr.unsqueeze(1), width=256, height=256, distance_scale=5.0
    )
    assert scores[0, 0] < 0.05


def test_overlap_score_nearby_facing_same_direction_higher_than_facing_away():
    # Query looks along +Y (yaw=0); candidates sit further along +Y (i.e.
    # inside the query's own view cone regardless of their own heading) so
    # visibility is satisfied for both and only direction_align differs.
    query = _pose_at(0.0, 0.0, 1.5).unsqueeze(0)
    same_dir = _pose_at(0.0, 1.0, 1.5).unsqueeze(0)
    opposite_dir = _pose_at(0.0, 1.0, 1.5, yaw=torch.pi).unsqueeze(0)
    intr = torch.tensor([[100.0, 100.0, 128.0, 128.0]])

    candidates = torch.stack([same_dir.squeeze(0), opposite_dir.squeeze(0)], dim=0).unsqueeze(0)  # (1,2,4,4)
    cand_intr = intr.unsqueeze(1).repeat(1, 2, 1)
    scores = frustum_overlap_scores(query, intr, candidates, cand_intr, width=256, height=256, distance_scale=5.0)
    assert scores[0, 0] > scores[0, 1]


def test_select_top_k_shape_and_order():
    torch.manual_seed(0)
    query = _pose_at(0.0, 0.0, 1.5).unsqueeze(0)
    intr = torch.tensor([[100.0, 100.0, 128.0, 128.0]])
    k_candidates = 5
    history = torch.stack([_pose_at(float(i), 0.0, 1.5) for i in range(k_candidates)], dim=0).unsqueeze(0)
    hist_intr = intr.unsqueeze(1).repeat(1, k_candidates, 1)

    idx, scores = select_top_k(query, intr, history, hist_intr, width=256, height=256, distance_scale=5.0, k=3)
    assert idx.shape == (1, 3)
    assert scores.shape == (1, 3)
    # Sorted descending.
    assert (scores[0, :-1] >= scores[0, 1:]).all()
    # Closest candidate (i=0, distance 0) should be first.
    assert idx[0, 0].item() == 0


def test_select_top_k_pads_when_fewer_candidates_than_k():
    query = _pose_at(0.0, 0.0, 1.5).unsqueeze(0)
    intr = torch.tensor([[100.0, 100.0, 128.0, 128.0]])
    history = _pose_at(1.0, 0.0, 1.5).unsqueeze(0).unsqueeze(1)  # (1,1,4,4)
    hist_intr = intr.unsqueeze(1)
    idx, scores = select_top_k(query, intr, history, hist_intr, width=256, height=256, distance_scale=5.0, k=4)
    assert idx.shape == (1, 4)
    assert (idx == 0).all()
