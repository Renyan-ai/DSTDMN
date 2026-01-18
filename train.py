import pandas as pd
import argparse
import time
import util
from util import *
import random
from model import DSTDMN
from ranger import Ranger
from logger_manager import setup_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import torch
import numpy as np
import signal
import sys

# -------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:1", help="")
parser.add_argument("--data", type=str, default="PEMS07", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--channels", type=int, default=128, help="number of nodes")
parser.add_argument("--num_nodes", type=int, default=307, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=12, help="out_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument("--epochs", type=int, default=1000, help="")
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument(
    "--save",
    type=str,
    default="./logs",
    help="save path (root log directory)",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="experiment name (if None, use timestamp)",
)
parser.add_argument(
    "--log_level",
    type=str,
    default="INFO",
    help="logging level (DEBUG, INFO, WARNING, ERROR)",
)
parser.add_argument(
    "--auto_clean_days",
    type=int,
    default=30,
    help="automatically archive logs older than this many days",
)
parser.add_argument(
    "--es_patience",
    type=int,
    default=100,
    help="quit if no improvement after this many iterations",
)
args = parser.parse_args()


class trainer:
    def __init__(self, scaler, input_dim, channels, num_nodes, input_len, output_len, dropout, lrate, wdecay, device):
        self.model = DSTDMN(device, input_dim, channels, num_nodes, input_len, output_len, dropout)
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5
        print("The number of parameters: {}".format(self.model.param_num()))
        print(self.model)

    def load_checkpoint(self, checkpoint_path, logger):
        if not os.path.exists(checkpoint_path):
            return False, 1, float('inf'), float('inf'), 0
        logger.info(f"正在从检查点 {checkpoint_path} 加载状态...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=next(self.model.parameters()).device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))
            best_test_mae = checkpoint.get('best_test_mae', float('inf'))
            epochs_since_best_mae = checkpoint.get('epochs_since_best_mae', 0)
            logger.info(f"继续从 Epoch {start_epoch + 1} 开始训练。")
            logger.info(f"上次最佳验证损失: {best_valid_loss:.4f}，上次最佳测试 MAE: {best_test_mae:.4f}")
            return True, start_epoch + 1, best_valid_loss, best_test_mae, epochs_since_best_mae
        except Exception as e:
            logger.error(f"❌ 加载检查点失败: {e}。将从头开始训练。")
            return False, 1, float('inf'), float('inf'), 0

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)



def save_checkpoint(epoch, model, optimizer, scheduler, valid_loss, test_mae, epochs_since_best_mae, checkpoint_path, logger):
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_valid_loss': valid_loss,
            'best_test_mae': test_mae,
            'epochs_since_best_mae': epochs_since_best_mae
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"✅ 检查点已保存到 {checkpoint_path}，当前 Epoch: {epoch}")
    except Exception as e:
        logger.error(f"❌ 保存检查点失败: {e}")


# -----------------------------


def main():
    seed_it(42)

    data = args.data

    # 生成实验名称（数据集名称 + 时间戳）
    if args.experiment_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{data}_{timestamp}"
    else:
        experiment_name = args.experiment_name

    # 初始化日志管理器
    logger_mgr = setup_logger(
        log_dir=args.save,
        experiment_name=experiment_name,
        log_level=args.log_level,
        auto_clean_days=args.auto_clean_days
    )

    logger = logger_mgr.logger

    # 定义全局变量，用于信号处理函数访问和修改 (注意：在Python 3中，信号处理器内部可以直接访问外部作用域的变量，但为了清晰，我们在这里定义它们)
    global current_epoch, loss, test_log, epochs_since_best_mae, engine, scheduler, checkpoint_path

    # 训练状态初始化（将在加载检查点后更新）
    start_epoch = 1
    loss = 9999999  # 对应 best_valid_loss
    test_log = 999999  # 对应 best_test_mae
    epochs_since_best_mae = 0
    bestid = 0  # 记录最佳模型的 epoch
    current_epoch = start_epoch

    name = args.data

    if args.data == "PEMS08":
        args.data = "data/" + args.data
        args.num_nodes = 170
        args.num_nodes = 207
    elif args.data == "PEMS08_60":
        args.data = "data/" + args.data
        args.num_nodes = 170
        args.input_len = 60
        args.output_len = 60
    elif args.data == "PEMS04_60":
        args.data = "data/" + args.data
        args.num_nodes = 307
        args.input_len = 60
        args.output_len = 60
    elif args.data == "PEMS03":
        args.data = "data/" + args.data
        args.num_nodes = 358
        args.epochs = 2000
        args.es_patience = 300
    elif args.data == "PEMS04":
        args.data = "data/" + args.data
        args.num_nodes = 307
    elif args.data == "PEMS07":
        args.data = "data/PEMS07"
        args.num_nodes = 883


    device = torch.device(args.device)

    # 保存实验配置
    logger_mgr.save_config(args)
    logger.info(f"实验配置: {vars(args)}")


    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
        )
    scaler = dataloader["scaler"]

    path = logger_mgr.get_model_save_path() + "/"  # 最佳模型保存路径
    checkpoint_path = os.path.join(logger_mgr.get_experiment_dir(),
                                   "checkpoint.pth")  # 检查点路径

    # 初始化训练器
    engine = trainer(
        scaler,
        args.input_dim,
        args.channels,
        args.num_nodes,
        args.input_len,
        args.output_len,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
    )
    # 初始化学习率调度器
    scheduler = ReduceLROnPlateau(
        engine.optimizer,
        mode='min',  # 监控指标是越小越好
        factor=0.5,  # 降低学习率的倍数 (每次减少 50%)
        patience=60,  # 连续 60 个 Epoch 验证损失不下降则触发
        verbose=True,
        min_lr=1e-6  # 最低学习率限制
    )

    # --- 检查点加载逻辑 ---
    # --- 检查点加载逻辑 (已修正) ---
    load_success, start_epoch_loaded, loss_loaded, test_log_loaded, epochs_since_best_mae_loaded = engine.load_checkpoint(
        checkpoint_path, logger)
    if load_success:
        start_epoch = start_epoch_loaded
        loss = loss_loaded
        test_log = test_log_loaded
        epochs_since_best_mae = epochs_since_best_mae_loaded
        bestid = start_epoch - 1
        try:
            checkpoint = torch.load(checkpoint_path)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            logger.warning(f"未能加载学习率调度器状态: {e}")

    current_epoch = start_epoch  # 关键：循环开始前设置

    # --- 信号处理函数 ---
    def signal_handler(sig, frame):
        logger.warning("🚨 收到中断信号 (Ctrl+C)。正在保存检查点...")
        save_checkpoint(
            current_epoch,
            engine.model,
            engine.optimizer,
            scheduler,
            loss,
            test_log,
            epochs_since_best_mae,
            checkpoint_path,
            logger
        )
        logger.info("程序安全退出。")
        sys.exit(0)

    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)
    logger.info(f"已设置 Ctrl+C (SIGINT) 信号处理。检查点路径: {checkpoint_path}")

    # -------------------

    # 记录模型信息
    logger_mgr.log_model_info(str(engine.model))

    logger.info("开始训练...")

    his_loss = []
    val_time = []
    train_time = []
    result = []
    test_result = []

    # 训练循环从 start_epoch 开始
    for epoch in range(start_epoch, args.epochs + 1):
        current_epoch = epoch  # 更新全局变量
        should_save_checkpoint = False
        logger_mgr.log_epoch_start(epoch, args.epochs)

        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []

        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainx = torch.Tensor(x).to(device)  # 64 12 170 1
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            # trainy[:, 0, :, :] 仅取第一个特征
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])

        t2 = time.time()
        train_time.append(t2 - t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])

        s2 = time.time()
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)

        scheduler.step(mvalid_loss)

        his_loss.append(mvalid_loss)
        train_m = dict(
            train_loss=mtrain_loss,
            train_rmse=mtrain_rmse,
            train_mape=mtrain_mape,
            train_wmape=mtrain_wmape,
            valid_loss=mvalid_loss,
            valid_rmse=mvalid_rmse,
            valid_mape=mvalid_mape,
            valid_wmape=mvalid_wmape,
        )
        train_m = pd.Series(train_m)
        result.append(train_m)

        # 记录epoch结束指标
        logger_mgr.log_epoch_end(
            epoch,
            {
                'loss': mtrain_loss,
                'rmse': mtrain_rmse,
                'mape': mtrain_mape,
                'wmape': mtrain_wmape,
            },
            {
                'loss': mvalid_loss,
                'rmse': mvalid_rmse,
                'mape': mvalid_mape,
                'wmape': mvalid_wmape,
            },
            train_time[-1],
            val_time[-1],
        )

        if mvalid_loss < loss:
            logger.info("###Update tasks appear###")
            if epoch < 100:
                loss = mvalid_loss
                torch.save(engine.model.state_dict(), path + "best_model.pth")
                bestid = epoch
                epochs_since_best_mae = 0
                logger_mgr.log_best_model(epoch, "Valid Loss", mvalid_loss)
                should_save_checkpoint = True

            elif epoch > 100:
                # === 测试集评估逻辑 (不变) ===
                outputs = []
                realy = torch.Tensor(dataloader["y_test"]).to(device)
                realy = realy.transpose(1, 3)[:, 0, :, :]

                for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
                    testx = torch.Tensor(x).to(device)
                    testx = testx.transpose(1, 3)
                    with torch.no_grad():
                        preds = engine.model(testx).transpose(1, 3)
                    outputs.append(preds.squeeze())

                yhat = torch.cat(outputs, dim=0)
                yhat = yhat[: realy.size(0), ...]

                amae = []
                amape = []
                awmape = []
                armse = []

                # 计算逐 horizon MAE
                for j in range(args.output_len):
                    pred = scaler.inverse_transform(yhat[:, :, j])
                    real = realy[:, :, j]
                    metrics = util.metric(pred, real)
                    amae.append(metrics[0])
                    amape.append(metrics[1])
                    armse.append(metrics[2])
                    awmape.append(metrics[3])

                avg_test_mae = np.mean(amae)

                logger.info(
                    f"Epoch {epoch} - Average Test MAE over {args.output_len} horizons: {avg_test_mae:.4f}")

                if avg_test_mae < test_log:
                    test_log = avg_test_mae
                    loss = mvalid_loss
                    torch.save(engine.model.state_dict(), path + "best_model.pth")
                    epochs_since_best_mae = 0
                    logger_mgr.log_best_model(epoch, "Test MAE (avg)", float(avg_test_mae))
                    bestid = epoch
                    logger.info(f"epoch: {epoch}")

                    # 记录逐 horizon 指标
                    for j in range(args.output_len):
                        logger.info(
                            f"Evaluate best model on test data for horizon {j + 1}, Test MAE: {amae[j]:.4f}, Test RMSE: {armse[j]:.4f}, Test MAPE: {amape[j]:.4f}, Test WMAPE: {awmape[j]:.4f}"
                        )
                    should_save_checkpoint = True  # <--- 标记保存检查点
                else:
                    epochs_since_best_mae += 1
                    logger.info("No update in Test MAE")

        else:
            epochs_since_best_mae += 1
            logger.info("No update in Valid Loss")

        # === 统一检查点保存逻辑 (实现您的要求) ===
        if should_save_checkpoint:
            save_checkpoint(
                current_epoch,
                engine.model,
                engine.optimizer,
                scheduler,
                loss,
                test_log,
                epochs_since_best_mae,
                checkpoint_path,
                logger
            )


        # === 200周期后每5个周期测试 ===
        # 原有逻辑：如果不是最佳模型，但满足 i >= 50 和 i % 5 == 0，执行一次测试集评估并记录
        if epoch >= 50 and epoch % 5 == 0 and not (mvalid_loss < loss and epoch > 100):
            outputs = []
            realy = torch.Tensor(dataloader["y_test"]).to(device)
            realy = realy.transpose(1, 3)[:, 0, :, :]

            for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                with torch.no_grad():
                    preds = engine.model(testx).transpose(1, 3)
                outputs.append(preds.squeeze())

            yhat = torch.cat(outputs, dim=0)
            yhat = yhat[: realy.size(0), ...]

            amae = []
            for j in range(args.output_len):
                pred = scaler.inverse_transform(yhat[:, :, j])
                real = realy[:, :, j]
                metrics = util.metric(pred, real)
                amae.append(metrics[0])  # 只收集MAE

            avg_test_mae = np.mean(amae)
            logger.info(
                f"Epoch {epoch} - Average Test MAE over {args.output_len} horizons: {avg_test_mae:.4f}")

        # 同步保存逐epoch指标快照
        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(
            f"{logger_mgr.get_experiment_dir()}/metrics/train.csv"
        )
        if epochs_since_best_mae >= args.es_patience and epoch >= 300:
            logger.info(f"早停触发：连续 {args.es_patience} 个 Epoch 性能未提升。")
            break

    # Output consumption
    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # test
    logger.info("Training ends")
    logger.info("The epoch of the best result：%s" % bestid)
    # his_loss 列表可能在加载检查点时有历史数据丢失，这里最好使用恢复的 loss 变量
    logger.info(
        "The valid loss of the best model %s" % str(round(loss, 4))
    )

    # ... (最终测试逻辑不变)
    engine.model.load_state_dict(torch.load(path + "best_model.pth"))
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    awmape = []

    test_m = []

    for i in range(args.output_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        logger.info(
            f"Evaluate best model on test data for horizon {i + 1}, Test MAE: {metrics[0]:.4f}, Test RMSE: {metrics[2]:.4f}, Test MAPE: {metrics[1]:.4f}, Test WMAPE: {metrics[3]:.4f}"
        )

        test_m = dict(
            test_loss=np.mean(metrics[0]),
            test_rmse=np.mean(metrics[2]),
            test_mape=np.mean(metrics[1]),
            test_wmape=np.mean(metrics[3]),
        )
        test_m = pd.Series(test_m)
        test_result.append(test_m)

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    logger.info(
        f"On average over {args.output_len} horizons, Test MAE: {np.mean(amae):.4f}, Test RMSE: {np.mean(armse):.4f}, Test MAPE: {np.mean(amape):.4f}, Test WMAPE: {np.mean(awmape):.4f}"
    )

    test_m = dict(
        test_loss=np.mean(amae),
        test_rmse=np.mean(armse),
        test_mape=np.mean(amape),
        test_wmape=np.mean(awmape),
    )
    test_m = pd.Series(test_m)
    test_result.append(test_m)

    # 输出测试结果至日志系统与CSV
    horizon_metrics = []
    for idx in range(len(amae)):
        horizon_metrics.append({
            'mae': float(amae[idx]),
            'rmse': float(armse[idx]),
            'mape': float(amape[idx]),
            'wmape': float(awmape[idx])
        })
    overall_metrics = {
        'mae': float(np.mean(amae)),
        'rmse': float(np.mean(armse)),
        'mape': float(np.mean(amape)),
        'wmape': float(np.mean(awmape))
    }
    logger_mgr.log_test_results(overall_metrics, horizon_metrics)

    test_csv = pd.DataFrame(test_result)
    test_csv.round(8).to_csv(
        f"{logger_mgr.get_experiment_dir()}/metrics/test.csv"
    )


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    # 程序总耗时记录
    print("Total time spent: {:.4f}".format(t2 - t1))
