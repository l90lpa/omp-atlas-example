ECKIT_INCLUDE_DIR = /home/azureuser/tools/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-11.4.0/eckit-1.25.2-6pdi45v5q374jmkhfm3jw77ybvgzwnk3/include
ATLAS_INCLUDE_DIR = /home/azureuser/projects/atlas-gpu-example/atlas/install/include
ATLAS_LIBRARY_DIR = /home/azureuser/projects/atlas-gpu-example/atlas/install/lib

all: main.cpp
	g++ -fopenmp -foffload=nvptx-none -fcf-protection=none -fno-stack-protector -no-pie -I$(ATLAS_INCLUDE_DIR) -I$(ECKIT_INCLUDE_DIR) -o app.exe main.cpp -L$(ATLAS_LIBRARY_DIR) -l atlas
