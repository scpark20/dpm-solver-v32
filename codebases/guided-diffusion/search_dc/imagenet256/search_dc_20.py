import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

N = 10
for scale in [2.0, 4.0, 6.0, 8.0]:
    dc_dir = f"/data/guided-diffusion/dc/imagenet256/scale{scale}"
    os.makedirs(dc_dir, exist_ok=True)

    file = f'samples/256x256_diffusion/dpmsolver++_1000_scale{scale}/images/samples_0.npz'
    data = np.load(file)
    traj = torch.tensor(data['hist_raw'])[:, :N]
    timesteps = torch.tensor(data['timesteps_raw'])
    classes = torch.tensor(data['classes'])
    print(traj.shape, timesteps.shape)

    import sys
    import torch
    import numpy as np

    from sample import parse_args_and_config, Diffusion

    #for NFE in [5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40]:
    for NFE in [5, 6]:
        for order in [3]:
            ###############################################################################
            # 1) Notebook에서 sys.argv를 직접 설정 (argparse 흉내)
            ###############################################################################
            sys.argv = [
                "sample.py",
                "--config", "imagenet256_guided.yml",  # 사용하려는 config
                "--sample_type", "dcsolver",
                "--timesteps", str(NFE),
                "--scale", str(scale),
                "--order", str(order),
                "--lower_order_final",
                "--dc_dir", dc_dir
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

            import torch.nn.functional as F
            traj = traj.to(device=diffusion.device)
            classes = classes.to(device=diffusion.device)
            with torch.no_grad():
                sampled_x, _ = diffusion.sample_image(traj[0], diffusion.model, classifier=diffusion.classifier, classes=classes, target=(traj, timesteps))
                print(f"NFE={NFE}, order={order}, loss={F.mse_loss(traj[-1], sampled_x)}")