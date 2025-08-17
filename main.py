import argparse
import os
import utils.models as models
import utils.datasets as datasets

from torch.utils.data import DataLoader
import torch
import random

from rci import VFL_RCI
from utils.utils import TempDataset
import torch.nn as nn
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='the datasets for evaluation;', type=str, choices=datasets.datasets_choices, default='yeast')
    parser.add_argument('--epochs', help='the number of epochs;', type=int, default=100)
    parser.add_argument('--batch_size', help='batch size;', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90])
    parser.add_argument('--emb_length', type=int, default=128)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--num_passive', help='number of passive parties', type=int, default=4)
    # 实验路径文件夹
    parser.add_argument('--expid', type=int, default=0)

    # 对比学习损失
    parser.add_argument('--use_con', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=0.07)
    
    # 微调 
    parser.add_argument('--lr_t', type=float, default=0.01)
    # 测试阶段缺失嵌入个数
    parser.add_argument('--k', type=int, default=1)

    # 噪声强度
    parser.add_argument('--lambda_d', type=float, default=0.1)
    
    # GMM 检测的阈值
    parser.add_argument('--use_gmm', type=int, default=0)
    parser.add_argument('--quantile', type=float, default=10)
    # Isolation Forest 检测的阈值
    parser.add_argument('--use_if', type=int, default=0)
    parser.add_argument('--contamination', type=float, default=0.05)
    # OCSVM 检测的阈值
    parser.add_argument('--use_ocsvm', type=int, default=0)
    parser.add_argument('--nu', type=float, default=0.05)
    
    args = parser.parse_args()


# python main.py --expid 0 --num_passive 4 --emb_length 128 --gpuid 0 --dataset yeast --use_con 1 --alpha 0.001 --k 1 --lambda_d 0.1 --use_gmm 1 --quantile 5
# python main.py --expid 0 --num_passive 4 --emb_length 128 --gpuid 0 --dataset yeast --use_con 1 --alpha 0.001 --k 1 --lambda_d 0.1 --use_if 1 --contamination 0.05
# python main.py --expid 0 --num_passive 4 --emb_length 128 --gpuid 0 --dataset yeast --use_con 1 --alpha 0.001 --k 1 --lambda_d 0.1 --use_ocsvm 1 --nu 0.05

    torch.cuda.set_device(args.gpuid)
    device = torch.device(f'cuda:{args.gpuid}')
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    
    dir = "/".join(os.path.abspath(__file__).split("/")[:-1])
    results_dir = os.path.join(dir, f"results_id={args.expid}", args.dataset)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    dataset_path = './dataset'
    if args.dataset == 'imagenet12': args.batch_size = 32
    
    if args.dataset in ['cifar10', 'mnist']:
        args.num_class = 10
    elif args.dataset in ['cifar100']:
        args.num_class = 100
    elif args.dataset in ['imagenet12']:
        args.num_class = 12
    elif args.dataset in ['yeast']:
        # args.num_class = 10
        args.num_class = 4
    elif args.dataset in ['letter']:
        args.num_class = 26

    data_train = datasets.datasets_dict[args.dataset](dataset_path, train=True)
    temp_dataset = TempDataset(full_dataset=data_train, transform=datasets.transforms_default[args.dataset])
    dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)

    data_test = datasets.datasets_dict[args.dataset](dataset_path, train=False)
    temp_dataset = TempDataset(full_dataset=data_test, transform=datasets.transforms_default[args.dataset])
    dataloader_test = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)

 
    entire_model_path = f'{results_dir}/entire_model_d={args.dataset}_n={args.num_passive}_emb={args.emb_length}_use_con={args.use_con}_alpha={args.alpha}.pth' 
    entire_model_tuned_path = f'{results_dir}/entire_model_d={args.dataset}_n={args.num_passive}_emb={args.emb_length}_use_con={args.use_con}_alpha={args.alpha}_tuned.pth' 

    results_model_log = open(f'{results_dir}/test_d={args.dataset}_n={args.num_passive}_emb={args.emb_length}_use_con={args.use_con}_alpha={args.alpha}.txt' ,'w')
    results_tuned_model_log = open(f'{results_dir}/test_d={args.dataset}_n={args.num_passive}_emb={args.emb_length}_use_con={args.use_con}_alpha={args.alpha}_tuned.txt' ,'w')

    if args.use_gmm:
        detection_similarity_log  = open(f'{results_dir}/detection_test_d={args.dataset}_n={args.num_passive}_emb={args.emb_length}_use_con={args.use_con}_alpha={args.alpha}_k={args.k}_lambda={args.lambda_d}_gmm_quantile={args.quantile}_similarity.txt','w')
    if args.use_if:
        detection_similarity_log  = open(f'{results_dir}/detection_test_d={args.dataset}_n={args.num_passive}_emb={args.emb_length}_use_con={args.use_con}_alpha={args.alpha}_k={args.k}_lambda={args.lambda_d}_if_contamination={args.contamination}_similarity.txt','w')
    if args.use_ocsvm:
        detection_similarity_log  = open(f'{results_dir}/detection_test_d={args.dataset}_n={args.num_passive}_emb={args.emb_length}_use_con={args.use_con}_alpha={args.alpha}_k={args.k}_lambda={args.lambda_d}_ocsvm_nu={args.nu}_similarity.txt','w')

    entire_model = models.entire[args.dataset](num_passive=args.num_passive, emb_length=args.emb_length)
    
    if os.path.exists(entire_model_path):
        entire_model = torch.load(entire_model_path, weights_only=False)
    entire_model = entire_model.to(device)
    
    
    #训练模型
    framework = VFL_RCI(args,entire_model,dataloader_train,dataloader_test,device)
    if not os.path.exists(entire_model_path):
        framework.train()
        torch.save(framework.model, entire_model_path)
    test_acc = framework.test()
    results_model_log.write('test_acc:%.2f\n'%(test_acc))
    results_model_log.flush()
    
    
    # 异常检测测试
    precision_similarity, recall_similarity, f1_similarity = framework.test_anamoly_detection(k=args.k, lambda_d=args.lambda_d)
    print('Precision:%.4f  Recall:%.4f  F1:%.4f\n'%(precision_similarity, recall_similarity, f1_similarity))
    detection_similarity_log.write('Precision:%.4f  Recall:%.4f  F1:%.4f\n'%(precision_similarity, recall_similarity, f1_similarity))


    # 微调模型
    if not os.path.exists(entire_model_tuned_path):
        framework.tune_active()
        torch.save(framework.model, entire_model_tuned_path)
    else:
        entire_model = torch.load(entire_model_tuned_path, weights_only=False)
        entire_model = entire_model.to(device)
    framework = VFL_RCI(args,entire_model,dataloader_train,dataloader_test,device)
    test_acc = framework.test()
    results_tuned_model_log.write('test_acc:%.2f\n'%(test_acc))
    results_tuned_model_log.flush()    
    
    
    # 嵌入缺失测试
    if args.num_passive == 2:
        for k in [1]:
            print(f'Offline number: {k}/4')
            results_recovery_log = open(f'{results_dir}/recovery_test_d={args.dataset}_n={args.num_passive}_emb={args.emb_length}_use_con={args.use_con}_alpha={args.alpha}_k={k}_tuned.txt','w')
            start = time.time()
            tuned_test_acc = framework.test_missing_embedding(num_to_mask=k)
            end = time.time()
            print(f"Dataset {args.dataset}  Elapsed time: {end - start:.6f} seconds")
            test_acc = framework.test()
            results_recovery_log.write('Normal Test Acc:%.2f Recovered Test Acc:%.2f  Elapsed time: {%.6f} seconds\n'%(test_acc, tuned_test_acc, end - start))
            results_recovery_log.flush()   
    
    if args.num_passive == 4:
        for k in [1,2,3]:
            print(f'Offline number: {k}/4')
            results_recovery_log = open(f'{results_dir}/recovery_test_d={args.dataset}_n={args.num_passive}_emb={args.emb_length}_use_con={args.use_con}_alpha={args.alpha}_k={k}_tuned.txt','w')
            start = time.time()
            tuned_test_acc = framework.test_missing_embedding(num_to_mask=k)
            end = time.time()
            print(f"Dataset {args.dataset}  Elapsed time: {end - start:.6f} seconds")
            test_acc = framework.test()
            results_recovery_log.write('Normal Test Acc:%.2f Recovered Test Acc:%.2f  Elapsed time: {%.6f} seconds\n'%(test_acc, tuned_test_acc, end - start))
            results_recovery_log.flush()   

    if args.num_passive == 7:
        for k in [1,2,3,4,5,6]:
            print(f'Offline number: {k}/8')
            results_recovery_log = open(f'{results_dir}/recovery_test_d={args.dataset}_n={args.num_passive}_emb={args.emb_length}_use_con={args.use_con}_alpha={args.alpha}_k={k}_tuned.txt','w')
            start = time.time()
            tuned_test_acc = framework.test_missing_embedding(num_to_mask=k)
            end = time.time()
            print(f"Dataset {args.dataset}  Elapsed time: {end - start:.6f} seconds")
            test_acc = framework.test()
            results_recovery_log.write('Normal Test Acc:%.2f Recovered Test Acc:%.2f  Elapsed time: {%.6f} seconds\n'%(test_acc, tuned_test_acc, end - start))
            results_recovery_log.flush()  

    if args.num_passive == 8:
        for k in [1,2,3,4,5,6,7]:
            print(f'Offline number: {k}/8')
            results_recovery_log = open(f'{results_dir}/recovery_test_d={args.dataset}_n={args.num_passive}_emb={args.emb_length}_use_con={args.use_con}_alpha={args.alpha}_k={k}_tuned.txt','w')
            start = time.time()
            tuned_test_acc = framework.test_missing_embedding(num_to_mask=k)
            end = time.time()
            print(f"Dataset {args.dataset}  Elapsed time: {end - start:.6f} seconds")
            test_acc = framework.test()
            results_recovery_log.write('Normal Test Acc:%.2f Recovered Test Acc:%.2f  Elapsed time: {%.6f} seconds\n'%(test_acc, tuned_test_acc, end - start))
            results_recovery_log.flush()  


if __name__ == '__main__':
    main()
