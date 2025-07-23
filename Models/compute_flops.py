import torch
from transformers import VivitForVideoClassification
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd
import os
import time

def analyze_model(model, input_shape, num_steps=10):
    """
    Analyzes the computational cost of a given model.

    Args:
        model: The model to be analyzed.
        input_shape: The shape of the input tensor (batch_size, num_frames, channels, height, width).
        num_steps: The number of steps for profiling.

    Returns:
        The profiler object and model parameter statistics.
    """
    # Count trainable and frozen parameters
    frozen_params = 0
    trainable_params = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
    
    total_params = trainable_params + frozen_params
    
    print("Model Parameter Analysis:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen params: {frozen_params:,}")
    
    # Create sample input
    device = next(model.parameters()).device
    inputs = torch.rand(*input_shape, device=device)
    
    # Set up optimizer - only optimizes trainable parameters
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    
    # Define training step
    def train_step():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = outputs.logits.mean()
        loss.backward()
        optimizer.step()
    
    # Warm-up
    for _ in range(3):
        train_step()
    
    # Set up profiler
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    
    # Profile the model
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True
    ) as prof:
        for _ in range(num_steps):
            with record_function("train_step"):
                train_step()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
    
    model_stats = {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "frozen_params": frozen_params
    }
    
    return prof, model_stats

def main():
    """
    Main function - compares the performance of different fine-tuning strategies.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")


    # To ensure a fair comparison, the classifier head is kept trainable in all strategies.
    configurations = [
        {
            "name": "Full_Finetuning",
            "layers_to_activate": "all" 
        },
        {
            "name": "Finetune_Encoder_Layer_0",
            "layers_to_activate": ["encoder.layer.0", "classifier"]
        }
    ]
    

    batch_size = 4
    input_shape = (batch_size, 32, 3, 224, 224)
    print(f"\nInput shape (batch_size={batch_size}): {input_shape}\n")

    all_results = []


    for config in configurations:
        print(f"--- Analyzing: {config['name']} ---")
        
        # Reload the model to ensure a consistent state for each analysis
        class_labels =['down','same','up']
        label2id = {label: i for i, label in enumerate(class_labels)}
        id2label = {i: label for label, i in label2id.items()}
        model_ckpt = "google/vivit-b-16x2-kinetics400"
        
        model = VivitForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        ).to(device)

        # Set parameter trainability based on the configuration
        if config["layers_to_activate"] == "all":
            for param in model.parameters():
                param.requires_grad = True
        else:

            for param in model.parameters():
                param.requires_grad = False

            for name, param in model.named_parameters():
                for layer_name in config["layers_to_activate"]:
                    if layer_name in name:
                        param.requires_grad = True
        

        prof, model_stats = analyze_model(model, input_shape)
        
        # Extract performance metrics
        flops = sum(event.flops for event in prof.key_averages() if hasattr(event, 'flops'))
        cpu_time_ms = sum(event.cpu_time for event in prof.key_averages()) / 1e6
        cuda_time_ms = sum(event.cuda_time for event in prof.key_averages() if hasattr(event, 'cuda_time')) / 1e6
        flops_step =flops / 10

        all_results.append({
            "Strategy": config['name'],
            "Trainable Params": model_stats['trainable_params'],
            "FLOPs (G)/step": flops_step / 1e9,
            "CPU Time (ms)": cpu_time_ms,
            "GPU Time (ms)": cuda_time_ms if cuda_time_ms > 0 else "N/A"
        })
        
        print(f"--- Analysis Complete: {config['name']} ---\n")

    results_df = pd.DataFrame(all_results)
    results_df.set_index("Strategy", inplace=True)
    
    print("\n--- Finetuning Strategy Performance Comparison ---")
    print(results_df.to_string())
    
    csv_filename = "finetuning_efficiency_comparison.csv"
    results_df.to_csv(csv_filename)
    print(f"\nComparison results saved to CSV: {os.path.abspath(csv_filename)}")


if __name__ == "__main__":
    main()