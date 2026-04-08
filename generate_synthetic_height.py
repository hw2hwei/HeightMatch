import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModelForDepthEstimation
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def round_to_nearest_14(x):
    return ((x + 13) // 14) * 14


def main(rank, world_size):
    parser = argparse.ArgumentParser(description='Depth Anything V2 with DDP')

    parser.add_argument('--imgdir', type=str, default='/home/Datasets/RSSeg/INRIA/Austin/image')
    parser.add_argument('--outdir', type=str, default='/home/Datasets/RSSeg/INRIA/Austin/height')

    args = parser.parse_args()

    # Initialize distributed processing
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Replace with master node's IP if distributed across nodes
    os.environ["MASTER_PORT"] = "29500"      # Replace with an unused port if necessary
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Process {rank} initialized.")

    device = torch.device(f'cuda:{rank}')
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").to(device)
    model = DDP(model, device_ids=[rank])

    # Define preprocessing
    rgb_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Gather filenames and distribute them across processes
    all_filenames = []
    for root, _, files in os.walk(args.imgdir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                all_filenames.append(os.path.join(root, file))

    all_filenames.sort()
    local_filenames = all_filenames[rank::world_size]  # Split files by rank

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for k, filename in enumerate(local_filenames):
        print(f"[Rank {rank}] Progress {k + 1}/{len(local_filenames)}: {filename}")

        # Read and preprocess image
        image = Image.open(filename).convert("RGB")
        raw_width, raw_height = image.size
        new_width, new_height = round_to_nearest_14(raw_width), round_to_nearest_14(raw_height)
        resized_image = image.resize((new_width, new_height), Image.BICUBIC)
        input_tensor = rgb_preprocess(resized_image).unsqueeze(0).to(device)

        # Infer depth
        with torch.no_grad():
            outputs = model(pixel_values=input_tensor)
        predicted_depth = outputs.predicted_depth.squeeze(0).detach().cpu().numpy()

        # Resize depth back to original resolution
        depth = Image.fromarray(predicted_depth).resize((raw_width, raw_height), Image.BICUBIC)
        
        # Convert depth to numpy array first
        depth_array = np.array(depth)
        
        # Normalize the depth values
        depth_array = ((depth_array - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
        
        # Convert back to Image
        depth = Image.fromarray(depth_array).convert("L")

        # Save output
        relative_path = os.path.relpath(filename, args.imgdir)
        output_path = os.path.join(args.outdir, os.path.splitext(relative_path)[0] + '.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        depth.save(output_path)

    # Clean up
    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()

    # Spawn distributed processes
    torch.multiprocessing.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )





