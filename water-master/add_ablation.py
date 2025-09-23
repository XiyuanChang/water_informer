import os
import shutil

# Define the source directory and the destination directory
source_dir = "ablation_small_donet"
destination_dir = "upload_deeponet_small_size"

# Define the file patterns to search for
file_patterns = [
    "ablation_small_donet/group_sbb/models/Mnist_LeNet/0726_052838/predictions/metric/metric.csv",
    "ablation_small_donet/group_sbb/models/Mnist_LeNet/0726_074358/predictions/metric/metric.csv",
    "ablation_small_donet/group_sbb/models/Mnist_LeNet/0726_095902/predictions/metric/metric.csv",
    "ablation_small_donet/group_sbb/models/Mnist_LeNet/0726_095901/predictions/metric/metric.csv",
    "ablation_small_donet/group_sbb/models/Mnist_LeNet/0726_121437/predictions/metric/metric.csv",
]

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Iterate through the source directory and find the files matching the patterns
for i in range(1, 17):  # Assuming 16 folders from group_1 to group_16
    for pattern in file_patterns:
        for root, _, files in os.walk(source_dir):
            for file in files:
                # print(pattern.replace("sbb", str(i)))
                if pattern.replace("sbb", str(i)) == os.path.join(root, file):
                    # Copy the file to the destination directory with the appropriate name
                    shutil.copyfile(
                        os.path.join(root, file),
                        os.path.join(destination_dir, f"group_{i}.csv"),
                    )
                    break