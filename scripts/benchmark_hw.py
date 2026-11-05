"""Measure train/infer throughput of candidate backbones on the local accelerator.
Usage: python scripts/benchmark_hw.py OUT.json model:res[:amp] ..."""
import time, json, sys, torch, timm

dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
out = {}
for spec in sys.argv[2:]:
    name, res, *amp = spec.split(":"); res = int(res); amp = bool(amp)
    m = timm.create_model(name, pretrained=False, num_classes=38).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-4); bs = 32
    x = torch.randn(bs, 3, res, res, device=dev); y = torch.randint(0, 38, (bs,), device=dev)
    m.train()
    for i in range(6):
        if i == 2: sync(); t = time.time()
        with torch.autocast(dev, dtype=torch.float16, enabled=amp):
            loss = torch.nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    sync(); tr = 4 * bs / (time.time() - t)
    m.eval()
    with torch.no_grad():
        for i in range(6):
            if i == 2: sync(); t = time.time()
            m(x)
    sync(); inf = 4 * bs / (time.time() - t)
    out[spec] = {"train_img_s": round(tr, 1), "infer_img_s": round(inf, 1), "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2)}
    print(spec, out[spec], flush=True)
    del m, opt
    if dev == "mps": torch.mps.empty_cache()
json.dump({"device": dev, "results": out}, open(sys.argv[1], "w"), indent=2)
