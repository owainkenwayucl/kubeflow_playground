
from kubeflow.training import TrainingClient

def train_pytorch():
    import torch
    import torch.nn
    import torchvision.models
    import torchvision.transforms
    import torch.distributed
    import medmnist
    import kubeflow.training.models
    import os

    num_epochs = 5

    dataset = "pathmnist"
    mlbc = "multi-label, binary-class"
    lr = 0.001

    print("Setting up medmnist")
    info = medmnist.INFO[dataset]
    task = info["task"]

    train_batch_size = 1024

    n_channels = info["n_channels"]
    n_classes = len(info["label"])

    data_class = getattr(medmnist, info["python_class"])

    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train = data_class(split="train", transform=data_transform, download=True, size=224, mmap_mode='r')

    train_dataloader = torch.utils.data.DataLoader(dataset=train, batch_size = train_batch_size, shuffle=True)

    print("medmnist set up")
    model = torchvision.models.resnet18(num_classes=n_classes)

    if task == mlbc:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    device, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")
    torch.distributed.init_process_group(backend=backend)

    local_rank = int(os.getenv("LOCAL_RANK", 0))
 
    print(f"Distributed Training with WORLD_SIZE: {torch.distributed.get_world_size()}, RANK: {torch.distributed.get_rank()}, LOCAL_RANK: {local_rank}.")

    device = torch.device(f"{device}:{local_rank}")
    model = torch.nn.parallel.DistributedDataParallel(model).to(device)
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimiser.zero_grad()
            outputs = model(inputs)

            if task == mlbc:
                targets = targets.to(torch.float32)
            else:
                targets = targets.squeeze().long()

            loss = criterion(outputs, targets)
            loss.backward()
            optimiser.step()
            if batch_idx % 10 == 0 and torch.distributed.get_rank() == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}")
        epoch_finish = time.time()
        
        timing["training"][f"epoch_{epoch}"] = epoch_finish - epoch_start

    timing["training"]["finish"] = time.time()
    timing["training"]["duration"] = timing["training"]["finish"] - timing["training"]["start"] 

    timing["inference"]["start"] = time.time()

    # Wait for the training to complete and destroy to PyTorch distributed process group.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print("Training is finished")
    torch.distributed.destroy_process_group()


jobid = "pathmnisttest"

tc = TrainingClient()

pathmnist_cache = kubeflow.training.models.V1Volume(name = "pathmnist_mount", persistent_volume_claim="pathmnist")
pathmnist_cache_m = kubeflow.training.models.V1VolumeMount(name="pathmnist_mount", mount_path="/root/.medmnist", read_only=True)

tc.create_job(
    name=jobid,
    train_func=train_pytorch,
    num_procs_per_worker="auto",
    num_workers=1,
    packages_to_install=['medmnist'],
    resources_per_worker={"gpu": "1"},
    volumes=[pathmnist_cache],
    volume_mounts=[pathmnist_cache_m]
)

print(tc.list_jobs())

print(jobid)
