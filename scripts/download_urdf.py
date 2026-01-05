"""
Download Kuka iiwa URDF models
"""

import os
import shutil
import pybullet_data


def download_kuka_urdf():
    """Download Kuka iiwa URDF from PyBullet data"""

    # PyBullet already includes URDF files
    # Just copy from pybullet_data package
    data_path = pybullet_data.getDataPath()
    kuka_path = os.path.join(data_path, "kuka_iiwa")

    print(f"PyBullet data path: {data_path}")
    print(f"Kuka URDF path: {kuka_path}")

    # Create target directory
    target_path = "./urdf/kuka_iiwa"
    os.makedirs("./urdf", exist_ok=True)

    if not os.path.exists(target_path):
        shutil.copytree(kuka_path, target_path)
        print(f"✓ Copied Kuka URDF to {target_path}")
    else:
        print(f"✓ Kuka URDF already exists at {target_path}")


if __name__ == "__main__":
    download_kuka_urdf()
