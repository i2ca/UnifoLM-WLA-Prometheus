import sys, time, numpy as np
sys.path.insert(0, ".")
from unifolm_wla.dataloader.multi_source_dataset.dataloader import create_training_dataloader
from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES
t = time.time()
dl, ds = create_training_dataloader("unifolm_wla/dataloader/multi_source_dataset/configs/prometheus_dex3.yaml",
                                    batch_size=2, num_workers=0, pin_memory=False, shuffle=True)
print("dataset:", len(ds), "amostras, montado em %.0f s" % (time.time() - t))
b = next(iter(dl))
ex = b if isinstance(b, dict) else b[0]
for k, v in (ex.items() if isinstance(ex, dict) else []):
    try:
        print(" ", k, getattr(v, "shape", type(v)), "" if not hasattr(v, "dtype") else v.dtype)
    except Exception:
        pass
m = ex["action_mask"][0].numpy()
ativos = [k for k, s in SLICES.items() if m[s].any()]
print("partes ativas da ação:", ativos)
a = ex["action"][0].numpy()
for k in ativos:
    print("  %-16s min %+.2f max %+.2f" % (k, a[:, SLICES[k]].min(), a[:, SLICES[k]].max()))
print("imagens:", {k: tuple(v.shape) for k, v in ex["images"].items()} if isinstance(ex["images"], dict) else type(ex["images"]))
print("tarefa:", ex["task"][0])
