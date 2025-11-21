# Introduction

![Diagram](/output.png)

This objective of this poject is to fine-tune the Stable Diffusion XL base model to generate images in the distinct art style of "Naruto" anime series.  
Because SDXL is a very large model—with across its UNet, base and refiner along with dual text encoders—the primary challenge is to accomplish this fine-tuning under **severe hardware constraints**, specifically:

- A **free-tier Google Colab** environment(available for 4-5 hours max in 24hrs window)
- **NVIDIA T4 GPU** with approximately **16GB VRAM**

To overcome these limitations, the project uses modern **parameter-efficient fine-tuning (PEFT)** methods and several GPU-optimization techniques that make it possible to train a high-quality art-style LoRA while staying within the Colab T4 limits.

---

# High-Level Overview of the Approach

The overall workflow consists of the following major steps:

1. **Dataset Preparation**  
   - Use the `lambdalabs/naruto-blip-captions` dataset from Hugging Face.  
   - Mount the output folder on googe drive to save the parameters
   - Used T4 GPU 16GB for training which allows only 4-5 hour of use

2. **SDXL LoRA Fine-Tuning**  
   - Use Hugging Face Diffusers’ official script `train_text_to_image_lora_sdxl.py`.  
   - Train only **LoRA adapters** instead of full SDXL weights to drastically reduce memory usage.  
   - Tried different set of training hyperparameters to make sure to be able to complete training without going out of memory.

3. **Optimization for 16GB VRAM**  
   - Train at **512 resolution** instead of 1024.  
   - Use **gradient checkpointing**, **mixed-precision**, and **activation checkpointing**.  
   - Reduce LoRA **rank** and batch sizes to fit GPU memory.  

4. **Checkpointing and Resume Strategy**  
   - Save LoRA weights every 100 steps to Google Drive.  
   - Resume training safely even if the Colab session disconnects.
   - For logging purposes, used **weights and biases**.

5. **Inference and Evaluation**  
   - Load LoRA on top of base SDXL.  
   - Perform **sequential inference** (unload base → load LoRA) to avoid memory issues since Colab would go out of memory if both are loaded simultaneously.  
   - Compare base vs LoRA outputs using the same prompt.

The combination of these techniques enables high-quality style learning while staying within Google Colab constraints.

# Techniques used

### LoRA

**What LoRA is:**  
LoRA (Low-Rank Adaptation) is a technique that adds small trainable “adapter” matrices inside a large model’s attention layers.  
Instead of updating the entire multi-billion-parameter SDXL model, LoRA trains only a tiny number of additional weights (often <1% of the model). LoRA lets us fine-tune SDXL for the Naruto art style efficiently, quickly, and within the strict memory limits of free Google Colab hardware

**Why we use LoRA here:**  
- **Massive VRAM savings** — only a few million parameters are trained instead of the full SDXL model.  
- **Much faster training** — fewer parameters means faster backprop and smaller optimizer state.  
- **Perfect for style learning** — LoRA is excellent for teaching a model a new *art style* without changing its core capabilities.  
- **Fits in Colab T4** — full SDXL fine-tuning is impossible on 16GB VRAM; LoRA makes it feasible.


### Increasing validation_epochs

**What `validation_epochs` does**  
`validation_epochs` controls **how often** the training script runs the validation (inference / sample generation) loop during training. Validation is useful for monitoring model progress and for saving sample images that show how the LoRA is evolving. \
However, on a constrained environment (Colab T4, 16GB VRAM) validation costs are non-trivial:

- Validation runs an inference pipeline which allocates large activations and can spike GPU memory usage
- Generating multiple validation images in one go increases peak memory and CPU/dataloader I/O.
- Increasing validation_epochs (i.e., validating less frequently) reduces how often these memory/compute spikes occur, which lowers the chance of OOM or runtime interruption. It also reduces the total time spent on inference during training, leaving more wall-clock time for optimization steps.
- `validation_epochs=1` → validate every epoch (default)
- `validation_epochs=5` → validate every 5 epochs (less frequent).


### Decreasing Resolution

**What changing `--resolution` does**  
The `--resolution` flag controls the size (height × width) of the training images fed into the SDXL UNet.  
By default, SDXL is trained at **1024×1024**, which is extremely VRAM-intensive. We used `validation_epochs=512` means every training image is resized (with random flip) to 512×512 before being passed into the network.

### Gradient Accumulation Steps

**What `gradient_accumulation_steps` does**   
`gradient_accumulation_steps=8` means the model runs **8 forward passes** (with batch size = 1) before doing **one backward update**. So although each step processes only 1 image, the *effective* batch size becomes 8.

   - Training with an actual batch size of 8 would cause an OOM on a T4.
   - Accumulation allows you to keep **batch_size=1** (very low memory) while still benefiting from a larger batch during optimization.
   - Larger batches reduce gradient noise and help LoRA training converge more reliably.
   - This is important for style LoRAs, which can become unstable with very small batches.

### Gradient Checkpointing

**What '--gradient_checkpointing' does:**  
`--gradient_checkpointing` tells the model **not to store all intermediate activations** during the forward pass. Instead, when the backward pass needs those activations, the model recomputes them on the fly. SDXL has a very large UNet with many attention blocks. Activations at resolution 512 or higher can take several gigabytes **per layer**. Checkpointing avoids storing most of them.

### Half Precision

**What it is:**  
'mixed_precision="fp16"` runs training in **half-precision (16-bit floats)** instead of full 32-bit precision. FP16 is a simple switch that provides huge VRAM savings and faster training — essential for running SDXL LoRA training on a T4 GPU.

**Why it's used:**  
- Cuts memory usage almost in half 
- Makes SDXL fit in 16GB VRAM  
- Speeds up training on Tensor Cores  
- Does not harm LoRA training quality  
- Enables gradient accumulation, validation, and larger resolution without OOM





  


