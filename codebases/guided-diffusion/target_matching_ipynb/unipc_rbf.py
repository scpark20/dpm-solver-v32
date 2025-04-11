import os
import torch
import numpy as np

noises = []
datas = []
classes = []
for i in range(10):
    file = f'samples/256x256_diffusion/unipc_200_scale0.0/images/samples_{i}.npz'
    if not os.path.exists(file):
        break
    data = np.load(file)
    noises.append(data['noises_raw'])
    datas.append(data['datas_raw'])
    classes.append(data['classes'])

noises = torch.tensor(np.concatenate(noises, axis=0))
datas = torch.tensor(np.concatenate(datas, axis=0))
classes = torch.tensor(np.concatenate(classes, axis=0))
print(noises.shape, datas.shape, classes.shape)

import sys
from sample import parse_args_and_config, Diffusion

for variant in ['bh1', 'bh2']:
    for steps in [5, 10, 15, 20, 25, 30, 35, 40]:
        for corrector_order in [2, 3]:
            ###############################################################################
            # 1) Notebook에서 sys.argv를 직접 설정 (argparse 흉내)
            ###############################################################################
            sys.argv = [
                "sample.py",
                "--config", "imagenet256_guided.yml",  # 사용하려는 config
                "--sample_type", "unipc_rbf",
                "--timesteps", str(steps),
                "--scale", "0.0",
                "--predictor_order", "2",
                "--corrector_order", str(corrector_order),
                "--variant", str(variant),
                "--lower_order_final",
                "--scale_dir", "/data/guided-diffusion/scale/unipc_rbf"
            ]
            ###############################################################################
            # 2) 인자/설정 로드
            ###############################################################################
            args, config = parse_args_and_config()

            ###############################################################################
            # 3) Diffusion 객체 생성 -> 모델 로딩
            ###############################################################################
            diffusion = Diffusion(args, config, rank=0)
            diffusion.prepare_model()

            indexes = np.random.randint(0, len(noises), size=(16,))
            print(indexes)
            noise_batch = noises[indexes].to(diffusion.device)
            target_batch = datas[indexes].to(diffusion.device)
            classes_batch = classes[indexes].to(diffusion.device)
            with torch.no_grad():
                sampled_x, _ = diffusion.sample_image(noise_batch, diffusion.model, classifier=diffusion.classifier, classes=classes_batch, target=target_batch)
