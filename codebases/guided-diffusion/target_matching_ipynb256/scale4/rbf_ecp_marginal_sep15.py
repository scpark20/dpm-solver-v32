import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


N = 128
M = 16
K = 20
for scale in [4.0]:
    os.makedirs(f"/data/guided-diffusion/scale/rbf_ecp_marginal_sep{scale}", exist_ok=True)
    noises = []
    datas = []
    classes = []
    for i in range(100):
        file = f'samples/256x256_diffusion/unipc_200_scale{scale}/images/samples_{i}.npz'
        if not os.path.exists(file):
            break
        data = np.load(file)
        noises.append(data['noises_raw'])
        datas.append(data['datas_raw'])
        classes.append(data['classes'])

    noises = torch.tensor(np.concatenate(noises, axis=0))[:N]
    datas = torch.tensor(np.concatenate(datas, axis=0))[:N]
    classes = torch.tensor(np.concatenate(classes, axis=0))[:N]
    print(noises.shape, datas.shape, classes.shape)

    import sys
    import torch
    import numpy as np

    from sample import parse_args_and_config, Diffusion

    for NFE in [15]:
        for order in [2, 3]:
            ###############################################################################
            # 1) Notebook에서 sys.argv를 직접 설정 (argparse 흉내)
            ###############################################################################
            sys.argv = [
                "sample.py",
                "--config", "imagenet256_guided.yml",  # 사용하려는 config
                "--sample_type", "rbf_ecp_marginal_sep",
                "--timesteps", str(NFE),
                "--scale", str(scale),
                "--order", str(order),
                "--lower_order_final",
                "--scale_dir", f"/data/guided-diffusion/scale/rbf_ecp_marginal_sep{scale}"
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

            for number in range(K):
                indexes = np.random.randint(0, len(noises), size=(M,))
                noise_batch = noises[indexes].to(device=diffusion.device)
                target_batch = datas[indexes].to(device=diffusion.device)
                classes_batch = classes[indexes].to(device=diffusion.device)
                
                with torch.no_grad():
                    sampled_x, _ = diffusion.sample_image(noise_batch, diffusion.model, classifier=diffusion.classifier, classes=classes_batch, target=target_batch, number=number)
                    print(f"NFE={NFE}, order={order}, loss={F.mse_loss(target_batch, sampled_x)}")
