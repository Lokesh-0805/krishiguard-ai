"""Per-step timing of the real training step to separate data-loading from GPU time."""
import sys, time, torch
sys.path.insert(0, "ml/training")
from common import *
from train import smoothed_weighted_ce
def main():
    dev = get_device(); torch.manual_seed(0)
    arch = sys.argv[1] if len(sys.argv) > 1 else "efficientnet_b0"
    df = load_split("train", 1600)
    dl = make_loader(df, train_transform(224), 32, True, int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    m = build_model(arch, 38).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-4); w = torch.ones(38, device=dev)
    m.train(); t_data = time.time()
    for i, (x, y) in enumerate(dl):
        t1 = time.time(); x, y = x.to(dev), y.to(dev)
        loss = smoothed_weighted_ce(m(x), y, w, 0.1); opt.zero_grad(); loss.backward(); opt.step(); torch.mps.synchronize()
        t2 = time.time()
        print(f"step {i}: data {t1-t_data:.2f}s  gpu {t2-t1:.2f}s", flush=True)
        t_data = time.time()
        if i == 25: break

if __name__ == "__main__":
    main()
