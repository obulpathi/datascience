# Instructions for setting up TensorFlow

# Base image: Ubuntu Server 14.04 LTS (HVM), SSD Volume Type - ami-9abea4fb
# Instance type: g2.2xlarge / g2.8xlarge
# Allocate 100 GB for root partition to make sure enough space for building drivers, tools and TensorFlow

# 1. Update and upgrade to make sure the AMI is upto date with security
sudo apt-get update
sudo apt-get upgrade -y

# 2. Install necessary packages
sudo apt-get install -y build-essential python-pip python-dev git python-numpy swig python-dev default-jdk zip zlib1g-dev

# 3. Blacklist opensource drivers for NVidia (nouveau drivers), which does not play well with NVidia GPUs
echo -e "blacklist nouveau\nblacklist lbm-nouveau\noptions nouveau modeset=0\nalias nouveau off\nalias lbm-nouveau off\n" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u
sudo reboot

# 4. To work aournd the drm issues, install the extra kernel modules
sudo apt-get install -y linux-image-extra-virtual
sudo reboot

# 5. We need kernel headers
sudo apt-get install -y linux-source linux-headers-`uname -r`

# 6. Install NVidia driver installers: Use only cuda 7.0
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run
chmod +x cuda_7.0.28_linux.run
./cuda_7.0.28_linux.run -extract=`pwd`/nvidia_installers
cd nvidia_installers

# 7. Install NVidia drivers
sudo ./NVIDIA-Linux-x86_64-346.46.run
# When prompted you'll need to:
# * Agree to the license terms
# * Accept the X Library path and X module path
# * Accept that 32-bit compatibility files will not be installed
# * Review and accept the libvdpau and libvdpau_trace libraries notice
# * Choose `Yes` when asked about automatically updating your X configuration file
# * Verify successful installation by choosing `OK`

# 8. Make sure nvidia drivers are properly installed by adding the nvidia kernel module
sudo modprobe nvidia

# 9. Install CUDA
sudo ./cuda-linux64-rel-7.0.28-19326674.run
# When prompted, you'll need to:
# * Accept the license terms (long scroll, page down with `f`)
# * Use the default installation path
# * Answer `n` to desktop shortcuts
# * Answer `y` to create a symbolic link


# 10. Install CUDNN nly use cudnn-6.5
wget http://developer.download.nvidia.com/compute/redist/cudnn/v2/cudnn-6.5-linux-x64-v2.tgz
tar -xzf cudnn-6.5-linux-x64-v2.tgz
sudo cp cudnn-6.5-linux-x64-v2/libcudnn* /usr/local/cuda/lib64
sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /usr/local/cuda/include/


# 11. We need Java 8 for building Bazel, which is required for building TensorFlow
sudo add-apt-repository ppa:openjdk-r/ppa
# When prompted you'll need to press ENTER to continue
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk

sudo update-alternatives --config java
sudo update-alternatives --config javac

# set JAVA_PATH, needed when building Bazel
export JAVA_HOME="/usr/lib/jvm/java-1.8.0-openjdk-amd64"

# 12. Install Bazel
cd
git clone https://github.com/bazelbuild/bazel.git
cd bazel
git checkout tags/0.1.4
./compile.sh
sudo cp output/bazel /usr/bin

# 13. Compile TensorFlow and build a Python package
cd
git clone --recurse-submodules https://github.com/tensorflow/tensorflow
git submodule update --init
git checkout tags/v0.8.0

bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install /tmp/tensorflow_pkg/tensorflow-0.8.0-cp27-none-linux_x86_64.whl


# 14. Test to see if TensorFlow is able to recognize the GPUs
cd ~/tensorflow/tensorflow/models/image/cifar10/
python cifar10_multi_gpu_train.py
# You should be able to see /gpu:0 on g2.2xlarge and /gpu:0-4 on g2.8xlarge

# References:
# * http://tleyden.github.io/blog/2014/10/25/cuda-6-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/
# * https://erikbern.com/2015/11/12/installing-tensorflow-on-aws/
