# RCI: Robust Vertical Federated Collaborative Inference

This repository contains the official implementation of the paper: **Robust Vertical Federated Inference for Multi-UAV Collaboration System**. In this paper, we propose RCI, a robust vertical federated collaborative inference framework for multi-UAV systems, which effectively tackles anomaly detection in VFL systems and mitigates inference failures arising from missing embeddings. 


## How to Run

1. **Download the dataset**  
   Place the required datasets into the `dataset/` directory.

2. **Perform backdoor detection**  
   You can evaluate detection performance using either method:
   ```bash
   python main.py --expid 0 --num_passive 4 --emb_length 128 --gpuid 0 --dataset mnist --use_con 1 --alpha 0.01 --k 1 --lambda_d 0.1 --use_if 1 --contamination 0.02
