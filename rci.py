import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from scipy.stats import norm
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import copy
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import time



def plot_feature_distribution(
    feature_values,
    true_labels,
    pred_labels,
    feature_name='distance',
    save_path='./',
    filename_prefix='anomaly_plot'
):
    """
    绘制某一指标的样本分布图（按真实标签和预测标签区分），并保存为 PDF。

    参数:
    - feature_values: numpy array, 每个样本的特征值（如距离或相似度）
    - true_labels: numpy array, 每个样本的真实标签（0=正常, 1=异常）
    - pred_labels: numpy array, 每个样本的预测标签（0=正常, 1=异常）
    - feature_name: str, 特征名称 ('distance' 或 'similarity')
    - save_path: str, 保存图像的路径
    - filename_prefix: str, 文件名前缀（自动拼接特征名）

    返回:
    - None
    """

    feature_values = np.array(feature_values)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    indices = np.arange(len(feature_values))

    plt.figure(figsize=(12, 5))

    # === 子图1：按真实标签分类 ===
    plt.subplot(1, 2, 1)
    plt.title(f'{feature_name.capitalize()} (Ground Truth)')
    plt.scatter(indices[true_labels == 0], feature_values[true_labels == 0], c='green', label='Benign', s=10, alpha=0.6)
    plt.scatter(indices[true_labels == 1], feature_values[true_labels == 1], c='red', label='Anomaly', s=10, alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel(feature_name.capitalize())
    plt.legend(loc='lower right')
    plt.grid(True)

    # === 子图2：按预测标签分类 ===
    plt.subplot(1, 2, 2)
    plt.title(f'{feature_name.capitalize()} (Prediction)')
    plt.scatter(indices[pred_labels == 0], feature_values[pred_labels == 0], c='green', label='Benign', s=10, alpha=0.6)
    plt.scatter(indices[pred_labels == 1], feature_values[pred_labels == 1], c='red', label='Anomaly', s=10, alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel(feature_name.capitalize())
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{filename_prefix}_{feature_name}.pdf")
    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"Feature distribution plot saved to: {save_file}")


def contrastive_loss_vfl(embeds, temperature=0.1):
    """
    embeds: Tensor of shape [N, K, D], where
      - N: number of samples
      - K: number of parties (embeddings per sample)
      - D: embedding dimension
    """
    N, K, D = embeds.shape
    device = embeds.device
    # embeds = F.normalize(embeds, dim=-1)  # cosine similarity

    # Flatten to [N*K, D]
    all_embeds = embeds.view(N * K, D)

    # Compute similarity matrix [N*K, N*K]
    sim_matrix = torch.matmul(all_embeds, all_embeds.T) / temperature

    # Mask: same sample, different party -> positive pairs
    sample_ids = torch.arange(N).unsqueeze(1).repeat(1, K).view(-1).to(device)  # shape [N*K]
    party_ids = torch.arange(K).repeat(N).to(device)  # shape [N*K]

    pos_mask = (sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1)) & \
               (party_ids.unsqueeze(0) != party_ids.unsqueeze(1))  # shape [N*K, N*K]

    # For each anchor, compute loss over its positives vs. all
    logits = sim_matrix
    logits_mask = ~torch.eye(N*K, dtype=torch.bool, device=device)  # mask self-similarity

    loss = []
    for i in range(N * K):
        # Only consider non-self similarities
        logits_i = logits[i][logits_mask[i]]  # shape [N*K - 1]
        pos_i = pos_mask[i][logits_mask[i]]   # shape [N*K - 1]

        if pos_i.sum() == 0:
            continue  # skip if no positive (shouldn’t happen if K ≥ 2)

        # softmax over all (positive + negatives)
        log_probs = F.log_softmax(logits_i, dim=0)
        loss_i = -log_probs[pos_i].mean()  # average over multiple positives
        loss.append(loss_i)

    return torch.stack(loss).mean()



# 异常检测-->解决嵌入异常问题
# --基于距离的异常检测
# --基于嵌入语义一致性的异常检测

# 模型微调-->解决嵌入缺失问题
# 缺失嵌入对抗微调






class VFL_RCI(object):
    def __init__(self, args, entire_model, train_loader, test_loader, device):
        self.args = args
        self.device = device
        
        self.model = entire_model
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.learning_rate = args.lr
        self.learning_rate_tuning = args.lr_t
        
        self.num_passive = args.num_passive
        self.epochs = 100

        self.use_con = args.use_con
        self.con_loss = contrastive_loss_vfl
        self.alpha = args.alpha  # 对比损失的权重
        self.temperature = args.temperature
        
        # GMM 检测的阈值
        self.use_gmm = args.use_gmm
        self.quantile = args.quantile
        # Isolation Forest 检测的阈值
        self.use_if = args.use_if
        self.contamination = args.contamination
        # OCSVM 检测的阈值
        self.use_ocsvm = args.use_ocsvm
        self.nu = args.nu
        
        # if self.attention_fuser is None:
        self.optimizer_active = torch.optim.Adam(self.model.active.parameters(), lr=self.learning_rate_tuning)
        self.scheduler_active = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_active,T_max=50,eta_min=1e-5)
        
        self.optimizer_entire = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.scheduler_entire = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_entire, milestones=[20, 40, 80], gamma=0.1)

    # VFL训练和测试相关
    def train_one(self, epoch):
        self.model.train()
        num_iter = (len(self.train_loader.dataset)//(self.args.batch_size))+1
        
        for batch_idx, (inputs, labels, indices) in enumerate(self.train_loader):
            if self.args.dataset == 'yeast' or self.args.dataset == 'letter':
                data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=1)]
            else:
                data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]
            labels = labels.to(self.device)
            
            embeddings = []
            for i in range(self.args.num_passive):
                tmp_emb = self.model.passive[i](data[i])
                embeddings.append(tmp_emb)

            agg_embeddings = self.model._aggregate(embeddings)
            
            logits = self.model.active(agg_embeddings)
            ce = F.cross_entropy(logits, labels)
            
            stacked_embeddings = torch.stack(embeddings, dim=1)
            if self.use_con:
                con_loss = self.con_loss(stacked_embeddings, temperature=self.temperature)
                loss = ce + self.alpha * con_loss
            else:
                loss = ce
            
            self.optimizer_entire.zero_grad()
            loss.backward()
            self.optimizer_entire.step()

            if batch_idx % 50 == 0:
                if self.use_con:
                    print('Dataset %s | Epoch [%3d/%3d]  Iter[%3d/%3d]  CE-loss: %.4f  CON-loss: %.4f\n'%(self.args.dataset, epoch, self.args.epochs, batch_idx+1, num_iter, ce.item(), self.alpha*con_loss.item()))
                else:
                    print('Dataset %s | Epoch [%3d/%3d]  Iter[%3d/%3d]  CE-loss: %.4f\n'%(self.args.dataset, epoch, self.args.epochs, batch_idx+1, num_iter, ce.item()))

        return

    def train(self,):
        print("\n============== Train VFL ==============")
        for epoch in range(self.args.epochs):
            self.train_one(epoch)
            self.test()
            self.scheduler_entire.step()

    def test(self):
        print("\n============== Test VFL ==============")
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        test_loss = 0
        correct_top1 = 0
        correct_top5 = 0  # 用于 cifar100
        num_samples = len(self.test_loader.dataset)

        with torch.no_grad():
            for i, (inputs, labels, indices) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                if self.args.dataset in ['yeast', 'letter']:
                    data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=1)]
                else:
                    data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]

                embeddings = []
                for i in range(self.args.num_passive):
                    tmp_emb = self.model.passive[i](data[i])
                    embeddings.append(tmp_emb)

                embeddings = torch.stack(embeddings, dim=1)
                agg_embeddings = embeddings.view(inputs.size(0), -1) 
                logits = self.model.active(agg_embeddings)
                losses = F.cross_entropy(logits, labels, reduction='none')
                test_loss += torch.sum(losses).item()

                if self.args.dataset == 'cifar100':
                    # Top-5 预测
                    _, top5_pred = logits.topk(5, dim=1, largest=True, sorted=True)
                    correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
                else:
                    # Top-1 预测
                    pred = logits.argmax(dim=1, keepdim=True)
                    correct_top1 += pred.eq(labels.view_as(pred)).sum().item()

        test_loss = test_loss / num_samples

        if self.args.dataset == 'cifar100':
            test_acc = 100. * correct_top5 / num_samples
            print('Test set: Average loss: {:.4f}, Top-5 Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct_top5, num_samples, test_acc))
        else:
            test_acc = 100. * correct_top1 / num_samples
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct_top1, num_samples, test_acc))

        return test_acc




    # 微调Top模型
    def tune_active(self):
        print("\n============== Top模型微调 ==============")
        
        self.model.active.train()
        num_iter = (len(self.train_loader.dataset) // self.args.batch_size) + 1

        # 初始化缓存列表，每个passive客户端对应一个字典
        embedding_cache = [{} for _ in range(self.args.num_passive)]

        for epoch in range(self.args.epochs // 2):
            for batch_idx, (inputs, labels, indices) in enumerate(self.train_loader):
                labels = labels.to(self.device)
                
                if epoch == 0:
                    # 只有第一个epoch计算并缓存嵌入
                    if self.args.dataset in ['yeast', 'letter']:
                        data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=1)]
                    else:
                        data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]

                    embeddings = []
                    for i in range(self.args.num_passive):
                        tmp_emb = self.model.passive[i](data[i])  # [B, D]
                        embeddings.append(tmp_emb)
                        # 缓存嵌入
                        for j, idx in enumerate(indices):
                            embedding_cache[i][idx.item()] = tmp_emb[j].detach()

                    embeddings = torch.stack(embeddings, dim=1)  # [B, num_passive, D]
                else:
                    # 之后直接用缓存的嵌入
                    B = inputs.size(0)
                    D = next(iter(embedding_cache[0].values())).shape[0]
                    embeddings = torch.zeros(B, self.args.num_passive, D, device=self.device)
                    for i in range(self.args.num_passive):
                        for j, idx in enumerate(indices):
                            embeddings[j, i] = embedding_cache[i][idx.item()]

                # === 构建随机掩码 ===
                num_to_mask = torch.randint(1, self.args.num_passive, (inputs.size(0),)).to(self.device)
                
                mask = torch.ones((inputs.size(0), self.args.num_passive), dtype=torch.bool).to(self.device)
                for index in range(inputs.size(0)):
                    masked_features = torch.randperm(self.args.num_passive)[:num_to_mask[index]]
                    mask[index, masked_features] = False

                # === 构建 masked_embeddings 和 full_embeddings ===
                masked_embeddings = embeddings.masked_fill(~mask.unsqueeze(-1), 0.0)  # 只 mask 掉部分

                masked_embeddings = masked_embeddings.view(inputs.size(0), -1)
                embeddings = embeddings.view(inputs.size(0), -1)

                logits_masked = self.model.active(masked_embeddings)
                logits_full = self.model.active(embeddings)

                # === 计算损失 ===
                ce_loss_masked = F.cross_entropy(logits_masked, labels)
                ce_loss_full   = F.cross_entropy(logits_full, labels)
                loss = ce_loss_masked + ce_loss_full
                
                self.optimizer_active.zero_grad()
                loss.backward()
                self.optimizer_active.step()

                sys.stdout.write(
                    f'Dataset {self.args.dataset} | Epoch [{epoch+1:3d}/{self.args.epochs//2}]  '
                    f'Iter[{batch_idx+1:3d}/{num_iter}] CE-Loss-Masked:{ce_loss_masked.item():.6f} CE-Loss-Full:{ce_loss_full.item():.6f} Loss: {loss.item():.6f}\n'
                )
                sys.stdout.flush()

            self.test()
            self.test_missing_embedding(num_to_mask=1)
            print('\n')
            self.scheduler_active.step()

    def test_missing_embedding(self, num_to_mask=1):
        print(f"\n============== Test Missing Embedding ==============")
        self.model.eval()
        self.model.active.eval()
        
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        correct_top1 = 0
        correct_top5 = 0
        num_samples = len(self.test_loader.dataset)

        with torch.no_grad():
            for index, (inputs, labels, _) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # === 数据切分 ===
                if self.args.dataset in ['yeast', 'letter']:
                    data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=1)]
                else:
                    data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]

                # === 获取 passive 端嵌入 ===
                emb_list = []
                for i in range(self.args.num_passive):
                    tmp_emb = self.model.passive[i](data[i])  # [B, D]
                    emb_list.append(tmp_emb)
                emb = torch.stack(emb_list, dim=1)  # [B, num_clients, D]

                # === 构造 mask ===
                mask = torch.ones((inputs.size(0), self.args.num_passive), dtype=torch.bool).to(self.device)
                for b in range(inputs.size(0)):
                    masked_features = torch.randperm(self.args.num_passive)[:num_to_mask]
                    mask[b, masked_features] = False

                emb = emb.masked_fill(~mask.unsqueeze(-1), 0.0)  # 将被 mask 的 client 嵌入置为全 0

                masked_emb = emb                    

                agg_emb = masked_emb.view(inputs.size(0), -1)

                logits = self.model.active(agg_emb)

                if self.args.dataset == 'cifar100':
                    _, top5_pred = logits.topk(5, dim=1, largest=True, sorted=True)
                    correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
                else:
                    pred = logits.argmax(dim=1, keepdim=True)
                    correct_top1 += pred.eq(labels.view_as(pred)).sum().item()

        # === 输出最终准确率 ===
        if self.args.dataset == 'cifar100':
            test_acc = 100. * correct_top5 / num_samples
            print('Recovered Embeddings Top-5 Accuracy: {}/{} ({:.2f}%)\n'.format(correct_top5, num_samples, test_acc))
        else:
            test_acc = 100. * correct_top1 / num_samples
            print('Recovered Embeddings Accuracy: {}/{} ({:.2f}%)\n'.format(correct_top1, num_samples, test_acc))

        return test_acc





    # 异常检测
    def test_anamoly_detection(self, k=1, lambda_d=0.1):
        similarities = self.compute_features()
        detector_similarity, threshold_similarity = self.train_detection_model_single(similarities)
        
        precision_similarity, recall_similarity, f1_similarity = self.evaluate_detection_model_single(detector_similarity, threshold_similarity, feature_name='similarity', k=k, lambda_d=lambda_d)
   
        return precision_similarity, recall_similarity, f1_similarity    


    def compute_features(self):
        similarities = []
        # 准备centroids_tensor，shape: [K, C, D]
        num_passive = self.args.num_passive

        for batch_idx, (inputs, labels, indices) in enumerate(self.train_loader):
            if self.args.dataset in ['yeast', 'letter']:
                data = [temp.to(self.device) for temp in torch.chunk(inputs, num_passive, dim=1)]
            else:
                data = [temp.to(self.device) for temp in torch.chunk(inputs, num_passive, dim=2)]

            labels = labels.cpu()
            B = inputs.size(0)
            K = num_passive

            embeddings = []
            for i in range(K):
                tmp_emb = self.model.passive[i](data[i])
                embeddings.append(tmp_emb.detach().cpu())  # [B,D]
            embeddings = torch.stack(embeddings, dim=1)  # [B,K,D]

            normed = F.normalize(embeddings, dim=2, p=2)
            sim_matrix = torch.bmm(normed, normed.transpose(1,2))  # [B,K,K]
            mask = torch.eye(K, device=sim_matrix.device).bool().unsqueeze(0)
            sim_matrix.masked_fill_(mask, 0)

            sim_sum = sim_matrix.sum(dim=2)  # [B,K]
            avg_sim = sim_sum / (K - 1)      # [B,K]
            similarities.append(avg_sim.reshape(-1,1))

        similarities = torch.cat(similarities, dim=0).numpy()

        return similarities


    # 单独指标的异常检测
    def train_detection_model_single(self, feature_array, feature_name='similarity'):
        feature_array = np.array(feature_array).reshape(-1, 1)

        if self.use_gmm:
            detector = GaussianMixture(n_components=1, covariance_type='full', random_state=0)
            detector.fit(feature_array)
            log_probs = detector.score_samples(feature_array)
            probs = np.exp(log_probs)
            threshold = np.percentile(probs, self.quantile)
            print(f"\n[GMM] {feature_name} threshold set at {self.quantile}th percentile: {threshold:.6f}")
            return detector, threshold

        elif self.use_if:
            detector = IsolationForest(contamination=self.contamination, random_state=0, n_estimators=3)
            detector.fit(feature_array)
            print(f"\n[Isolation Forest] {feature_name} model trained.")
            return detector, None  # 无需手动阈值

        elif self.use_ocsvm:
            detector = OneClassSVM(nu=self.nu, kernel='rbf', gamma='scale')
            detector.fit(feature_array)
            print(f"\n[One-Class SVM] {feature_name} model trained.")
            return detector, None

        else:
            raise ValueError("No valid detector type selected.")



    # 没有实现推理时间优化 
    def evaluate_detection_model_single(self, detector, threshold, feature_name='similarity', k=1, lambda_d=0.1):
        assert feature_name in ['distance', 'similarity'], "feature_name must be 'distance' or 'similarity'"
        print(f"====== Evaluate Anomaly Detection using {feature_name} only ======")

        all_preds_flat = []
        all_labels_flat = []
        feature_values = []

        
        for batch_idx, (inputs, labels, _) in enumerate(self.test_loader):
            if self.args.dataset in ['yeast', 'letter']:
                data_chunks = torch.chunk(inputs, self.args.num_passive, dim=1)
            else:
                data_chunks = torch.chunk(inputs, self.args.num_passive, dim=2)

            data_chunks = [d.to(self.device) for d in data_chunks]
            labels = labels.to(self.device)
            B = inputs.size(0)
            K = self.args.num_passive

            with torch.no_grad():
                embeddings = [self.model.passive[i](data_chunks[i]) for i in range(K)]

            all_indices = np.arange(B)
            np.random.shuffle(all_indices)
            num_anomaly = B // 2
            anomaly_indices = set(all_indices[:num_anomaly])

            for j in anomaly_indices:
                corrupted = torch.randperm(K)[:k]
                for idx in corrupted:
                    dim = embeddings[idx].size(1)
                    embeddings[idx][j] = embeddings[idx][j] + lambda_d * torch.randn(dim).to(self.device)

            embeddings_full = torch.stack(embeddings, dim=1)

            for j in range(B):
                if feature_name == 'similarity':
                    normed = torch.nn.functional.normalize(embeddings_full[j], dim=1)
                    sim_matrix = torch.matmul(normed, normed.T)
                    mask = torch.eye(K, dtype=torch.bool, device=sim_matrix.device)
                    sim_matrix.masked_fill_(mask, 0.0)
                    avg_sim = sim_matrix.sum(dim=1) / (K - 1)
                    feature_value = avg_sim.min().item()

                feature = np.array([[feature_value]], dtype=np.float64)

                # === 检测 ===
                if self.use_gmm:
                    prob = np.exp(detector.score_samples(feature)[0])
                    is_anomaly = int(prob < threshold)
                else:
                    pred = detector.predict(feature)[0]  # 1正常 -1异常
                    is_anomaly = int(pred == -1)

                sample_label = int(j in anomaly_indices)
                all_preds_flat.append(is_anomaly)
                all_labels_flat.append(sample_label)
                feature_values.append(feature_value)

        # === 评估 ===
        precision = precision_score(all_labels_flat, all_preds_flat)
        recall = recall_score(all_labels_flat, all_preds_flat)
        f1 = f1_score(all_labels_flat, all_preds_flat)

        # method = "GMM" if self.use_gmm else "IForest" if self.use_if else "OCSVM"
        # print(f"[{method} - {feature_name}] Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}\n")

        # 可视化
        # feature_values = np.array(feature_values)
        # all_labels_flat = np.array(all_labels_flat)

        # if self.use_gmm:
        #     fig_prefix = f'{method.lower()}_d={self.args.dataset}_n={self.args.num_passive}_emb={self.args.emb_length}_use_con={self.args.use_con}_alpha={self.args.alpha}_k={k}_lambda_d={lambda_d}_quantile={self.args.quantile}_sigle'
        # else:
        #     fig_prefix = f'{method.lower()}_d={self.args.dataset}_n={self.args.num_passive}_emb={self.args.emb_length}_use_con={self.args.use_con}_alpha={self.args.alpha}_k={k}_lambda_d={lambda_d}_sigle'

        # plot_feature_distribution(feature_values=feature_values, true_labels=all_labels_flat, pred_labels=all_preds_flat, feature_name=feature_name, save_path='plots', filename_prefix=fig_prefix)

        return precision, recall, f1



    # def evaluate_detection_model_single(self, detector, threshold, feature_name='similarity', k=1, lambda_d=0.1):
    #     assert feature_name in ['distance', 'similarity'], "feature_name must be 'distance' or 'similarity'"
    #     print(f"====== Evaluate Anomaly Detection using {feature_name} only ======")

    #     all_features_list = []
    #     all_labels_list = []

    #     K = self.args.num_passive
    #     mask = torch.eye(K, dtype=torch.bool, device=self.device).unsqueeze(0)  # (1, K, K)

    #     for batch_idx, (inputs, labels, _) in enumerate(self.test_loader):
    #         if self.args.dataset in ['yeast', 'letter']:
    #             data_chunks = torch.chunk(inputs, K, dim=1)
    #         else:
    #             data_chunks = torch.chunk(inputs, K, dim=2)

    #         data_chunks = [d.to(self.device, non_blocking=True) for d in data_chunks]
    #         B = inputs.size(0)
    #         labels = labels.to(self.device, non_blocking=True)

    #         with torch.no_grad():
    #             embeddings = [self.model.passive[i](data_chunks[i]) for i in range(K)]
    #         embeddings_full = torch.stack(embeddings, dim=1)  # (B, K, D)

    #         perm = torch.randperm(B, device=self.device)
    #         num_anom = B // 2
    #         anomaly_idx = perm[:num_anom]
    #         anomaly_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
    #         anomaly_mask[anomaly_idx] = True

    #         if k > 0 and num_anom > 0:
    #             rand_scores = torch.rand((num_anom, K), device=self.device)
    #             topk_idx = torch.topk(rand_scores, k, dim=1).indices  # (num_anom, k)

    #             rows = anomaly_idx.unsqueeze(1).expand(-1, k).reshape(-1)
    #             cols = topk_idx.reshape(-1)

    #             corruption_mask = torch.zeros((B, K), dtype=torch.bool, device=self.device)
    #             corruption_mask[rows, cols] = True

    #             D = embeddings_full.size(2)
    #             num_selected = rows.numel()
    #             if num_selected > 0:
    #                 noise_vals = lambda_d * torch.randn((num_selected, D), device=self.device, dtype=embeddings_full.dtype)
    #                 embeddings_full[rows, cols] += noise_vals

    #         if feature_name == 'similarity':
    #             normed = torch.nn.functional.normalize(embeddings_full, dim=2)  # (B, K, D)
    #             sim_matrix = torch.bmm(normed, normed.transpose(1, 2))  # (B, K, K)
    #             sim_matrix.masked_fill_(mask, 0.0)
    #             avg_sim = sim_matrix.sum(dim=2) / (K - 1)  # (B, K)
    #             feat_batch = avg_sim.min(dim=1).values.unsqueeze(1)  # (B, 1)
    #         else:
    #             raise NotImplementedError(f"Feature '{feature_name}' not implemented")

    #         # 保留在 GPU，统一放进列表
    #         all_features_list.append(feat_batch)
    #         all_labels_list.append(anomaly_mask.long())

    #     # 批量合并，统一拷贝到 CPU
    #     features_all = torch.cat(all_features_list, dim=0).cpu().numpy()
    #     labels_all = torch.cat(all_labels_list, dim=0).cpu().numpy()

    #     # sklearn 检测器批量推理（保持不变）
    #     if self.use_gmm:
    #         probs = np.exp(detector.score_samples(features_all))
    #         preds = (probs < threshold).astype(int)
    #     else:
    #         preds = (detector.predict(features_all) == -1).astype(int)

    #     precision = precision_score(labels_all, preds)
    #     recall = recall_score(labels_all, preds)
    #     f1 = f1_score(labels_all, preds)

    #     return precision, recall, f1
