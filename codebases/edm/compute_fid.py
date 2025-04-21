import tensorflow_gan as tfgan
import tensorflow as tf
import numpy as np
import os

from evaluation import *
import gc
from tqdm import tqdm


inception_model = get_inception_model(inceptionv3=False)
BATCH_SIZE = 1000


def load_cifar10_stats():
    """Load the pre-computed dataset statistics."""
    filename = "/data/checkpoints/cifar10_stats.npz"

    with tf.io.gfile.GFile(filename, "rb") as fin:
        stats = np.load(fin)
        return stats


def compute_fid(path):
    data_stats = load_cifar10_stats()
    images = []
    for file in os.listdir(path):
        if file.endswith(".npz"):
            with tf.io.gfile.GFile(os.path.join(path, file), "rb") as fin:
                sample = np.load(fin)
        images.append(sample["samples"])
    samples = np.concatenate(images, axis=0)
    all_pools = []
    N = samples.shape[0]
    if N < 50000:
        return None
    for i in tqdm(range(N // BATCH_SIZE)):
        gc.collect()
        latents = run_inception_distributed(
            samples[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, ...], inception_model, inceptionv3=False
        )
        gc.collect()
        all_pools.append(latents["pool_3"])
    all_pools = np.concatenate(all_pools, axis=0)[:50000, ...]
    data_pools = data_stats["pool_3"]
    fid = tfgan.eval.frechet_classifier_distance_from_activations(data_pools, all_pools)
    return fid

names = ['dpm_solver++', "uni_pc_bh1", 'dpm_solver_v3', 'rbf_ecp_marginal_2.0_3', 'rbf_ecp_marginal_2.0_4', 'rbf_ecp_marginal_3.0_3', 'rbf_ecp_marginal_2.0_4']
steps = [5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40]
if not os.path.isdir('fid'):
    os.makedirs('fid')
for step in steps:    
    for name in names:
        # 각 name과 step에 해당하는 파일명을 만들고, 결과를 기록
        filename = f"{name}_{step}_output.txt"
        filename = os.path.join('fid', filename)
        if os.path.exists(filename):
            continue
        path = f"/data/edm/{name}_{step}"
        if not os.path.exists(path):
            continue
        fid = compute_fid(path)  # FID 계산
        if fid is None:
            continue
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"name : {name}\n")
            f.write(f"step : {step}\n")
            f.write(f"FID  : {fid}\n")

        print(f"파일 생성 완료: {filename} (FID={fid})")


# for name in ["dpm_solver++", "heun", "uni_pc_bh1", "uni_pc_bh2", "dpm_solver_v3"]:
#     fids = []
#     for step in [5, 6, 8, 10, 12, 15, 20, 25]:
#         path = f"samples/edm-cifar10-32x32-uncond-vp/{name}_{step}"
#         fid = compute_fid(path)
#         fids.append(float(fid))
#     print(name, fids, file=open("output.txt", "a"))

