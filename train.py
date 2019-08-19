from __future__ import division

from models import *
#from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
import test
import visdom_plots as vplt 
#from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from visdom import Visdom
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch") # changed to 1 from 8
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/detext.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=6, help="number of cpu threads to use during batch generation") # changed to 6 from 8
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension") #changed to 50 from 416
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=False, help="allow for multi-scale training") # changed to false
    parser.add_argument("--log", default=datetime.datetime.now().strftime("%d%b%Y_%H%M"), help="set log folder name and visdom plots set environment name") # changed to false
    opt = parser.parse_args()
    print(opt)
    log_name=opt.log
    #logger = Logger("logs")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    #create directory to store logs
    log_dir = "logs/log_{}".format(log_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # initialiase visdom plotting and required variables
    plots = vplt.VisdomLinePlotter(log_name)
    #use to delete environement  plots.viz.delete_env("13Jul2019_0946")
    plotting_batch = 0
    
    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    train_label_path = data_config["train_label"]
    valid_path = data_config["valid"]
    valid_label_path = data_config["valid_label"]
    class_names = load_classes(data_config["names"])
    
    #store the arguments
    with open(log_dir+"/parameters.txt", 'w+') as f:
                    f.write(str(opt))
    #f= open("logs/Log_{}.txt".format(log_name),"w+")
    #f.write(opt)
    #f.close(())
    
    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, train_label_path, img_size=opt.img_size, augment=True, multiscale=opt.multiscale_training)# augment = changed to false
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        batch_log = []
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            #print("batch"+str(batch_i))
            #print(targets)
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            log_str += "| ------------|-------------|-------------|------------ |\n"
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            save_log = [["Epoch", "Batch", "Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            txt_log = []
            txt_log += [("epoch", epoch)]
            txt_log += [("batch", batch_i)]
            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]
                save_log += [[epoch, batch_i, metric, *row_metrics]]
                # Tensorboard logging
                #tensorboard_log = [] 
                #for j, yolo in enumerate(model.yolo_layers):
                 #   for name, metric in yolo.metrics.items():
                  #      if name != "grid_size":
                   #         tensorboard_log += [(f"{name}_{j+1}", metric)]
                            #txt_log += [(f"{name}_{j+1}", metric)]
                #tensorboard_log += [("loss", loss.item())]
                
            for indx, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            txt_log += [(f"{name}_{indx+1}", metric)]
                            #plots.plot(f"{name}", f"{name}_YOLO_layer_{indx+1}", f"{name}_YOLO_layer_{indx+1}", plotting_batch, metric)                
                            
            txt_log += [("loss", loss.item())]
            batch_log += txt_log
            with open(log_dir+"/batch_data.txt", 'a') as f:
                f.write(str(txt_log)+'\n')
                txt_log = []
            for i, val in enumerate(metric_table):
                valj = "| "
                if i == 1:
                    valj += "------------|-------------|-------------|------------ |\n| "
                for j, val2 in enumerate(val):
                    if len(val[j]) <= 12:
                        for k in range(0,12-len(val[j])):
                            val[j] += " "
                    val[j] += "| "
                    valj += val[j]
                valj +="\n"
                log_str += valj
            log_str += "| ------------|-------------|-------------|------------ |\n"
            # log_str += metric_table 
            #print(metric_table)
            #log_str += metric_table
            log_str += f"\nTotal loss {loss.item()}"
            #np.savetxt("logs/)
            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)
            # ====== plotting loss of the class per batch==================
            
            plots.plot("Loss Batch", "Train", "Training Batch Loss", plotting_batch, loss.item(), "Batches")
            plotting_batch += 1
        
        #========== plotting mean batch metrics for the epoch ================
        if epoch % opt.evaluation_interval == 0:
            batch_stats = defaultdict(list)
            for i, name in enumerate(batch_log):
                batch_stats[name[0]].append(name[1])    
            for k in range(3):        
                plots.plot("Loss_YOLO", "Loss_YOLO_{}".format(k+1), "Training Loss at YOLO Layers", epoch, np.mean(batch_stats["loss_{}".format(k+1)]), "Epochs")            
                plots.plot("x_YOLO", "x_YOLO_{}".format(k+1), "Training x at YOLO Layers", epoch, np.mean(batch_stats["x_{}".format(k+1)]), "Epochs")
                plots.plot("y_YOLO", "y_YOLO_{}".format(k+1), "Training y at YOLO Layers", epoch, np.mean(batch_stats["y_{}".format(k+1)]), "Epochs")
                plots.plot("w_YOLO", "w_YOLO_{}".format(k+1), "Training w at YOLO Layers", epoch, np.mean(batch_stats["w_{}".format(k+1)]), "Epochs")
                plots.plot("h_YOLO", "h_YOLO_{}".format(k+1), "Training h at YOLO Layers", epoch, np.mean(batch_stats["h_{}".format(k+1)]), "Epochs")
                plots.plot("conf_YOLO", "conf_YOLO_{}".format(k+1), "Training conf at YOLO Layers", epoch, np.mean(batch_stats["conf_{}".format(k+1)]), "Epochs")
                plots.plot("cls_YOLO", "cls_YOLO_{}".format(k+1), "Training cls at YOLO Layers", epoch, np.mean(batch_stats["cls_{}".format(k+1)]), "Epochs")
                plots.plot("cls_acc_YOLO", "cls_acc_YOLO_{}".format(k+1), "Training cls_acc at YOLO Layers", epoch, np.mean(batch_stats["cls_acc_{}".format(k+1)]), "Epochs")
                plots.plot("recall50_YOLO", "recall50_YOLO_{}".format(k+1), "Training recall50 at YOLO Layers", epoch, np.mean(batch_stats["recall50_{}".format(k+1)]), "Epochs")
                plots.plot("precision_YOLO", "precision_YOLO_{}".format(k+1), "Training precision at YOLO Layers", epoch, np.mean(batch_stats["precision_{}".format(k+1)]), "Epochs")
                plots.plot("conf_obj_YOLO", "conf_obj_YOLO_{}".format(k+1), "Training conf_obj at YOLO Layers", epoch, np.mean(batch_stats["conf_obj_{}".format(k+1)]), "Epochs")
                plots.plot("conf_noobj_YOLO", "conf_noobj_YOLO_{}".format(k+1), "Training conf_noobj at YOLO Layers", epoch, np.mean(batch_stats["conf_noobj_{}".format(k+1)]), "Epochs")
            plots.plot("epoch_loss","Loss", "Loss", epoch, np.mean(batch_stats["loss"]), "Epochs")
        if epoch == 0:
            plots.plot("mAP_Epoch", "Mean AP", "Mean AP", epoch, 0, "Epochs")
        #========= evaluation ==================    
        if epoch % opt.evaluation_interval == 0 and epoch != 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = test.evaluate(
                model,
                im_path=valid_path,
                im_gt_path=valid_label_path,
                iou_thres=0.5,
                conf_thres=0.5, #changed from 0.5 to 0.2
                nms_thres=0.5, #changed from 0.5 to 0.2
                img_size=opt.img_size,
                batch_size=opt.batch_size,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            #logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class Name", "AP"]]
            txt_log_epoch = []
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                txt_log_epoch += (("epoch", epoch), ("class_no", c),("class_name", class_names[c]),("AP", "%.5f" % AP[i]))
                
            with open(log_dir+"/epoch_data.txt", 'a') as f:
                    f.write(str(txt_log_epoch)+'\n')
                    txt_log_epoch = []
            epoch_log_str = "| ------------|-------------|------------ |\n"
            for j, val in enumerate(ap_table):
                valk = "| "
                if j == 1:
                    valk += "------------|-------------|------------ |\n| "
                for k, val2 in enumerate(val):
                    if len(str(val[k])) <= 12:
                        for m in range(0,12-len(str(val[k]))):
                            val[k] = str(val[k])
                            val[k] += " "
                    val[k] += "| "
                    valk += val[k]
                valk +="\n"
                epoch_log_str += valk
            epoch_log_str += "| ------------|-------------|------------ |\n"
            print(epoch_log_str)
            print(f"---- mAP {AP.mean()}")
            #======= plotting mean AP for the epoch ============
            plots.plot("mAP_Epoch", "Mean AP", "Mean AP", epoch, AP.mean(), "Epochs")
            plots.plot("Recall_Epoch", "Recall", "Recall", epoch, recall.mean(), "Epochs")
            plots.plot("Precision_Epoch", "Precision", "Precision", epoch, precision.mean(), "Epochs")
            plots.plot("f1_Epoch", "f1 Measure", "f1 Measure", epoch, f1.mean(), "Epochs")
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
