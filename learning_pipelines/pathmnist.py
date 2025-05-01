# Naïve attempt convert the training loop from https://github.com/owainkenwayucl/ML_Playground/tree/main/MedMNIST_PL/General
# This is a terrible way to do it and it needs to be re-arched.

# Needs two PVCs - one to cache dataset ("pathmnist") and one to hold our output ("medmnistcheckpoints")
from kfp import dsl, compiler, kubernetes

@dsl.component(base_image="nvcr.io/nvidia/pytorch:25.03-py3", 
               packages_to_install=['lightning','medmnist','onnx','onnxscript','onnxruntime'])
def training(d_num_epochs:int, d_repeats:int, d_batch_size:int, d_base:str) -> str:
    import numpy
    import torch
    import torch.nn
    import torch.utils.data
    import torch.optim

    import warnings

    # Suppress Torchvision warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    import torchvision.models
    import torchvision.transforms
    warnings.filterwarnings("default", category=UserWarning)

    import tqdm

    import time
    import json
    import sys
    import os

    import pytorch_lightning

    import medmnist

    import onnx
    import onnxruntime

    mlbc = "multi-label, binary-class"

    def detect_platform():
        num_acc = "auto"
        device = "auto"
        # Set up devcies
        if torch.cuda.is_available():
            device = "cuda"
            num_acc = torch.cuda.device_count()
            deviceid = 0
            for i in range(num_acc):
                device_name = torch.cuda.get_device_name(i)
                print(f"Detected Cuda Device: {device_name}")
                freemem,_ = torch.cuda.mem_get_info(i)
                print(f"Free memory: {freemem}")
                if freemem > 1024:
                    deviceid = i
                    
            torch.set_float32_matmul_precision('high')
            print("Enabling TensorFloat32 cores.")
            device = f"cuda:{deviceid}"
        # can't use multiprocessing in kubernetes containers
        num_acc = 1

        return device, num_acc


    def generate_dataloaders(dataset, batch_size):
        info = medmnist.INFO[dataset]
        task = info["task"]

        n_channels = info["n_channels"]
        n_classes = len(info["label"])

        data_class = getattr(medmnist, info["python_class"])

        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        train = data_class(split="train", transform=data_transform, download=True, size=224, mmap_mode='r')
        val = test = data_class(split="val", transform=data_transform, download=True, size=224, mmap_mode='r')
        test = data_class(split="test", transform=data_transform, download=True, size=224, mmap_mode='r')

        # can't use multiprocessing in kubernetes containers
        train_dataloader = torch.utils.data.DataLoader(dataset=train, batch_size = batch_size, shuffle=True, num_workers=0, persistent_workers=False)
        val_dataloader = torch.utils.data.DataLoader(dataset=val, batch_size = batch_size, shuffle=False, num_workers=0, persistent_workers=False)
        test_dataloader = torch.utils.data.DataLoader(dataset=test, batch_size = batch_size, shuffle=False, num_workers=0, persistent_workers=False)

        print(train)

        return train_dataloader, val_dataloader, test_dataloader, task, n_classes

    class Resnet_Classifier(pytorch_lightning.LightningModule):
        def __init__(self, device, task, lr, base_model):
            super().__init__()
            self.model = base_model
            self.task = task
            self.device_name = device
            self.log_safe = True
            self.lr = lr

            if task == mlbc:
                self.loss_module = torch.nn.BCEWithLogitsLoss()
            else:
                self.loss_module = torch.nn.CrossEntropyLoss()

        def forward(self, images):
            return self.model(images)

        def training_step(self, batch, batch_idx):
            inputs, targets = batch
            outputs = self.model(inputs)

            if self.task == mlbc:
                targets = targets.to(torch.float32)
            else:
                targets = targets.squeeze().long()

            loss = self.loss_module(outputs, targets)

            accuracy = (outputs.argmax(dim=-1) == targets).float().mean()
            if self.log_safe:
                self.log("train_acc", accuracy, on_step=False, on_epoch=True, sync_dist=True)
                self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            inputs, targets = batch
            outputs = self.model(inputs)

            if self.task == mlbc:
                targets = targets.to(torch.float32)
            else:
                targets = targets.squeeze().long()

            loss = self.loss_module(outputs, targets)

            accuracy = (outputs.argmax(dim=-1) == targets).float().mean()
            if self.log_safe:
                self.log("val_acc", accuracy, sync_dist=True)
            else:
                self.val_outputs.append(accuracy)
            return loss

        def test_step(self, batch, batch_idx):
            inputs, targets = batch
            outputs = self.model(inputs)

            if self.task == mlbc:
                targets = targets.to(torch.float32)
            else:
                targets = targets.squeeze().long()

            loss = self.loss_module(outputs, targets)

            accuracy = (outputs.argmax(dim=-1) == targets).float().mean()
            if self.log_safe:
                self.log("test_acc", accuracy, sync_dist=True)
            else:
                self.test_outputs.append(accuracy)
            return loss

        def on_validation_epoch_end(self) -> None:
            if not self.log_safe:
                if str(self.device) == "cpu":
                    self.log("val_acc", torch.stack(self.val_outputs).mean())
                self.val_outputs.clear()

        def on_test_epoch_end(self) -> None:
            if not self.log_safe:
                if str(self.device) == "cpu":
                    self.log("test_acc", torch.stack(self.test_outputs).mean())
                self.test_outputs.clear()

        def configure_optimizers(self):
            optimiser = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
            return optimiser

    def write_onnx(model, filename):
        gibberish = torch.randn(1, 3, 224, 224, requires_grad=True)
        model_cpu = model.to("cpu")
        model_cpu.eval()
        torch_gibberish = model_cpu(gibberish)
        onnx_file = filename
        onnx_out_model = torch.onnx.export(model_cpu, 
                                    gibberish,
                                    onnx_file,
                                    export_params = True,
                                    input_names = ['input'],                       # the model's input names
                                    output_names = ['output'],                     # the model's output names
                                    dynamic_axes = {'input' : {0 : 'batch_size'},    # variable length axes
                                                    'output' : {0 : 'batch_size'}})

        print("Checking with ONNX")

        onnx_model = onnx.load(onnx_file)

        print("Checking with ONNX Runtime")


        ort_session = onnxruntime.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(gibberish)}
        ort_outs = ort_session.run(None, ort_inputs)

        # compare ONNX Runtime and PyTorch results
        numpy.testing.assert_allclose(to_numpy(torch_gibberish), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Validation success")

    def main(num_epochs=5, repeats=1, batch_size=1024, base="resnet18", lr=0.001):
        cwd = os.getcwd()
        print(f"Initial location: {cwd}")
        nwd = f"/root/pathmnist"
        if not os.path.exists(nwd):
            os.makedirs(nwd)
        os.chdir(nwd)
        cwd = os.getcwd()
        print(f"Current location: {cwd}")
        print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

        device, num_acc = detect_platform()
        print(f"Detected device config: {device}:{num_acc}")
        stats = {}
        # Define parameters
        dataset = "pathmnist"

        train_dl, val_dl, test_dl, task, num_classes = generate_dataloaders(dataset, batch_size)

        base_model = torchvision.models.resnet18(num_classes=num_classes)
        base_model_str = base

        if base != None:
            if base == "resnet34":
                base_model_str = "resnet34"
                base_model = torchvision.models.resnet34(num_classes=num_classes)
            elif base == "resnet50":
                base_model_str = "resnet50"
                base_model = torchvision.models.resnet50(num_classes=num_classes)
            elif base == "resnet101":
                base_model_str = "resnet101"            
                base_model = torchvision.models.resnet101(num_classes=num_classes)
            elif base == "resnet152":
                base_model_str = "resnet152"
                base_model = torchvision.models.resnet152(num_classes=num_classes)
            elif base == "wideresnet50":
                base_model_str = "wideresnet50_2"
                base_model = torchvision.models.wide_resnet50_2(num_classes=num_classes)
            elif base == "wideresnet101":
                base_model_str = "wideresnet101_2"
                base_model = torchvision.models.wide_resnet101_2(num_classes=num_classes)
            elif base == "vgg11":
                base_model_str = "vgg11"
                base_model = torchvision.models.vgg11(num_classes=num_classes)

        model = Resnet_Classifier(device, task, lr, base_model)

        prec_words = "32bit"

        for repeat in range(repeats):
            corrected_epochs = num_epochs * (1 + repeat)
            trainer = pytorch_lightning.Trainer(max_epochs=num_epochs, accelerator=device, devices=num_acc)
                
            print(f"Performing training iteration {repeat + 1} of {repeats} for {corrected_epochs} epochs.")

            output_filename = f"medmnist_classifier_{base_model_str}_{dataset}_{corrected_epochs}_{repeats}_{prec_words}"

            checkpoint_filename = f"{output_filename}.ckpt"
            onnx_filename = f"{output_filename}.onnx"
            weights_filename = f"{output_filename}.weights"
            json_filename = f"{output_filename}.json"

            stats["output_filename"] = output_filename
            stats["checkpoint_filename"] = checkpoint_filename
            stats["onnx_filename"] = onnx_filename
            stats["weights_filename"] = weights_filename
            stats["json_filename"] = json_filename
            stats["device"] = device
            stats["num_accelerators"] = num_acc
            stats["num_epochs"] = corrected_epochs
            stats["repeats"] = repeats
            stats["batch_size"] = batch_size
            stats["lr"] = lr
            stats["precision"] = prec_words

            print(json.dumps(stats, indent=4))

            trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

            val_model = model
            val_trainer = trainer
            val_stats = val_trainer.validate(model=val_model, dataloaders=val_dl)
            test_stats = val_trainer.test(model=val_model, dataloaders=test_dl)

            stats["validation_stats"] = val_stats
            stats["test_stats"] = test_stats

            trainer.save_checkpoint(checkpoint_filename)
            torch.save(model.model.state_dict(), weights_filename)

            if trainer.global_rank == 0:
                write_onnx(model=val_model, filename=onnx_filename)

                log_data = json.dumps(stats, indent=4)
                print(log_data)

                with open(json_filename, "w") as lfh:
                    lfh.write(log_data)

    main(num_epochs=d_num_epochs, repeats=d_repeats, batch_size=d_batch_size, base=d_base)
    return("complete")

@dsl.pipeline
def pathmnist_pipeline(num_epochs:int, repeats:int, batch_size:int, base:str) -> str:
    gpu_task = training(d_num_epochs=num_epochs, d_repeats=repeats, d_batch_size=batch_size, d_base=base).set_memory_request("80Gi").add_node_selector_constraint(accelerator="nvidia.com/gpu").set_accelerator_limit(1)
    kubernetes.mount_pvc(
        gpu_task,
        pvc_name='medmnistcheckpoints',
        mount_path='/root/pathmnist',
    )

    kubernetes.mount_pvc(
        gpu_task,
        pvc_name='pathmnist',
        mount_path='/root/.medmnist',
    )


    return gpu_task.output

compiler.Compiler().compile(pathmnist_pipeline, 'pathmnist_pipeline.yaml')
