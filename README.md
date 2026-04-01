# train_mnist.py

# MNIST Klassifizierung mit PyTorch

Dieses Projekt implementiert ein einfaches Multi-Layer Perceptron (MLP) zur Erkennung handgeschriebener Ziffern (MNIST).

## Technische Details
* **Framework**: PyTorch.
* **Datenquelle**: MNIST (via Torchvision).
* **Architektur**: 
    * Input: 784 Neuronen.
    * Hidden Layers: 2x Linear (128 Neuronen) mit ReLU.
    * Output: 10 Neuronen mit Softmax.
* **Hardware**: Automatische CUDA-Erkennung für GPU-Beschleunigung.

## Installation und Ausführung
1. Erforderliche Bibliotheken installieren:
   `pip install torch torchvision numpy`
2. Skript starten:
   `python train_mnist.py`

## Parameter (args)
Die Konfiguration erfolgt zentral über ein Dictionary im Skript:
* Batch Size: 20.
* Learning Rate: 1e-4.
* Epochs: 30.
