from kfp import dsl, compiler
from kfp.client import Client

@dsl.component(base_image="nvcr.io/nvidia/pytorch:25.03-py3")
def count_gpus() -> str:
    import torch
    num_gpus = torch.cuda.device_count()
    r = f"Detected {torch.cuda.device_count()} Cuda devices."
    for a in range(torch.cuda.device_count()):
        r = r + f"Detected Cuda Device {a}: {torch.cuda.get_device_name(a)}"
    print(r)
    return r

@dsl.pipeline
def gpu_pipeline() -> str:
    gpu_task = count_gpus().set_gpu_limit(1)
    return gpu_task.output

compiler.Compiler().compile(gpu_pipeline, 'gpu_pipeline.yaml')

client = Client()
run = client.create_run_from_pipeline_package(
    'gpu_pipeline.yaml',
    arguments={},
)