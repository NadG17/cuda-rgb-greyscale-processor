# **CUDA RGB to Grayscale Image Processor**

This project uses **NVIDIA CUDA** to convert RGB images to grayscale, leveraging **GPU acceleration** for high-performance image processing. It runs smoothly on **Google Colab**, automatically handling all dependencies and image generation steps.

## **🚀 Features**

⚡ **Fast grayscale conversion** using CUDA  
🧪 **Synthetic image generation** (gradients, circles, patterns)  
📊 **Performance benchmarking**  
✅ **Runs directly on Google Colab** – no local CUDA setup required  

## **🛠️ Setup Guide**

### **1️⃣ Launch Google Colab**
- Go to [Google Colab](https://colab.research.google.com/)
- Click **New Notebook**

### **2️⃣ Enable GPU Runtime**
- Click **Runtime** → **Change runtime type**
- Set **Hardware accelerator** to **GPU**
- Preferably select **T4** or **V100**
- Click **Save**

### **3️⃣ Verify GPU Access**
Paste this in the first code cell, it should execute in about 30-40 seconds:

```python
!nvidia-smi
print("\n" + "="*50)
print("✅ GPU is available and ready!")
```

## **▶️ Running the Processor**

### **4️⃣ Clone and Run the Project**
Paste this into a new code cell:

```bash
# Clone repo and execute
!git clone https://github.com/NadG17/cuda-rgb-greyscale-processor.git
%cd cuda-rgb-greyscale-processor
!chmod +x run.sh && ./run.sh
```

**This will:**
- Clone the GitHub repo
- Build the CUDA `.cu` file using `nvcc`
- Install required dependencies (OpenCV etc.)
- Generate **over 20 synthetic RGB test images**
- Process them using **two CUDA kernel versions**
- Save grayscale outputs
- Generate a **performance report** and **execution logs**

## **🖼️ View Image Results**

To visualize original and grayscale images side-by-side:

```python
import matplotlib.pyplot as plt
import cv2
import glob
import os
from math import ceil

input_images = sorted(glob.glob('random_color_images/*.jpg'))
total_images = len(input_images)

if total_images == 0:
    print("No images found!")
else:
    cols = min(6, total_images)
    rows = ceil(total_images / cols) * 2
    fig = plt.figure(figsize=(4 * cols, 6 * ceil(total_images / cols)))
    
    for idx, img_path in enumerate(input_images):
        row_pair = idx // cols
        col = idx % cols
        
        # Original
        orig = cv2.imread(img_path)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        ax1 = plt.subplot(rows, cols, row_pair * 2 * cols + col + 1)
        ax1.imshow(orig_rgb)
        ax1.set_title(f"Original\n{os.path.basename(img_path)}", fontsize=9)
        ax1.axis('off')
        
        # Grayscale
        filename = os.path.splitext(os.path.basename(img_path))[0]
        gray_path = f'random_greyscaled_images/{filename}_greyscale.jpg'
        ax2 = plt.subplot(rows, cols, (row_pair * 2 + 1) * cols + col + 1)
        
        if os.path.exists(gray_path):
            gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            ax2.imshow(gray, cmap='gray')
            ax2.set_title(f"Grayscale\n{os.path.basename(gray_path)}", fontsize=9)
        else:
            ax2.text(0.5, 0.5, "Not Found", ha='center', va='center')
            ax2.set_title("Missing")
        ax2.axis('off')

    plt.tight_layout()
    plt.show()
```

## **📊 Analyze Results**

Paste this code to inspect processing statistics:

```python
import os

print("📈 DETAILED PROCESSING ANALYSIS")
print("="*60)

input_count = len([f for f in os.listdir('random_color_images') if f.endswith('.jpg')])
output_count = len([f for f in os.listdir('random_greyscaled_images') if f.endswith('.jpg')])

print(f"📁 Input Images: {input_count}")
print(f"📁 Grayscale Outputs: {output_count}")
print(f"✅ Success Rate: {(output_count / input_count) * 100:.1f}%")

def get_dir_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)

input_size = get_dir_size('random_color_images')
output_size = get_dir_size('random_greyscaled_images')

print(f"💾 Input Size: {input_size:.2f} MB")
print(f"💾 Output Size: {output_size:.2f} MB")
print(f"🗜️ Compression Ratio: {input_size / output_size:.2f}x")

print("\n📋 SAMPLE GENERATED FILES:")
for filename in sorted(os.listdir('random_color_images'))[:10]:
    print(f"   • {filename}")
if input_count > 10:
    print(f"   ... and {input_count - 10} more")
```

## **🧩 Project Structure**

```
cuda-rgb-greyscale-processor/
├── comparison_images/            # Benchmark comparison outputs
├── random_color_images/          # Generated input images
├── random_greyscaled_images/     # Grayscale outputs
├── convertRGBToGrey.cu           # CUDA kernel code
├── convertRGBToGrey.hpp          # Header file
├── Makefile                      # Build configuration
├── run.sh                        # Automated script
├── README.md                     # You're reading it!
├── execution_log.txt             # GPU execution logs
└── performance_report.md         # Summary of benchmarks
```
