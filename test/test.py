import torch

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/build/"))
)

import minigrad


def almost_equal(a, b, tol=1e-6):
    return abs(a - b) < tol


def run_test():
    ta = torch.tensor([2.0], requires_grad=True)
    tb = torch.tensor([3.0], requires_grad=True)

    tc = ta + tb
    td = ta - tb

    te = tc * td

    tf = torch.tanh(te)
    tg = tf**3
    th = torch.relu(te)

    va = minigrad.Value(2)
    vb = minigrad.Value(3)

    vc = va + vb
    vd = va - vb

    ve = vc * vd

    vf = ve.tanh()
    vg = vf.pow(3)
    vh = ve.relu()

    assert almost_equal(
        ta.item(), va.getData()
    ), f"Mismatch: {ta.item()} vs {va.getData()}"
    assert almost_equal(
        tb.item(), vb.getData()
    ), f"Mismatch: {tb.item()} vs {vb.getData()}"

    assert almost_equal(
        tc.item(), vc.getData()
    ), f"Mismatch: {tc.item()} vs {vc.getData()}"
    assert almost_equal(
        td.item(), vd.getData()
    ), f"Mismatch: {td.item()} vs {vd.getData()}"

    assert almost_equal(
        te.item(), ve.getData()
    ), f"Mismatch: {te.item()} vs {ve.getData()}"
    assert almost_equal(
        tf.item(), vf.getData()
    ), f"Mismatch: {tf.item()} vs {vf.getData()}"
    assert almost_equal(
        tg.item(), vg.getData()
    ), f"Mismatch: {tg.item()} vs {vg.getData()}"
    assert almost_equal(
        th.item(), vh.getData()
    ), f"Mismatch: {th.item()} vs {vh.getData()}"

    print("All MiniGrad values match PyTorch (within tolerance)!")


if __name__ == "__main__":
    run_test()
