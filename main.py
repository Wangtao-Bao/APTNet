from torch import optim, export
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.data import *
from utils.metric import *
from argparse import ArgumentParser
from loss import *
import time
from tqdm import tqdm
from APTNet import APTNet
from tool import *
os.environ['CUDA_VISIBLE_DEVICES']="0"

def parse_args():

    # Setting parameters
    parser = ArgumentParser(description='Implement of model')
    parser.add_argument('--dataset-dir', type=str, default='dataset/NUAA-SIRST')
    #parser.add_argument('--dataset-dir', type=str, default='dataset/IRSTD-1K')
    parser.add_argument("--save_img_dataset", default='NUAA-SIRST', type=str, help="dataset name")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--warm-epoch', type=int, default=5)#热身轮数
    parser.add_argument('--base-size', type=int, default=320)
    parser.add_argument('--crop-size', type=int, default=320)
    parser.add_argument("--save_folder", default='weightNUAA/APTNet-%s' % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))))
    parser.add_argument('--multi-gpus', type=bool, default=False)
    parser.add_argument('--if-checkpoint', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--weight_path', type=str, default='weightNUAA/weight.pkl')
    parser.add_argument("--save_img", default=False, type=bool, help="save image of or not")
    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test'

        self.args = args
        self.start_epoch = 0   
        self.mode = args.mode

        trainset = IRSTD_Dataset(args, mode='train',name=args.save_img_dataset)
        valset = IRSTD_Dataset(args, mode='val',name=args.save_img_dataset)

        self.train_loader = Data.DataLoader(trainset, args.batch_size, shuffle=True, drop_last=True,num_workers=8,pin_memory=True)
        self.val_loader = Data.DataLoader(valset, 1, drop_last=False,num_workers=8,pin_memory=True)

        device = torch.device('cuda')
        self.device = device

        model=APTNet(input_channels=3)


        if args.multi_gpus:
            if torch.cuda.device_count() > 1:
                print('use '+str(torch.cuda.device_count())+' gpus')
                model = nn.DataParallel(model, device_ids=[0, 1])

        model.to(device)
        self.model = model

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1500, eta_min=1e-5)

        self.down = nn.MaxPool2d(2, 2)
        self.loss_fun = SLSIoULoss()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC = ROCMetric(1, 10)
        self.best_iou = 0
        self.warm_epoch = args.warm_epoch


        if args.mode=='train':
            if args.if_checkpoint:
                check_folder = 'weightNUAA/APTNet-2024-07-07-10-05-07'
                checkpoint = torch.load(check_folder+'/checkpoint.pkl')
                self.model.load_state_dict(checkpoint['net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch']+1
                self.best_iou = checkpoint['iou']
                self.save_folder = check_folder
            else:
                self.save_folder = args.save_folder
                if not osp.exists(self.save_folder):
                    os.mkdir(self.save_folder)
        if args.mode == 'test':
            weight = torch.load(args.weight_path)
            self.model.load_state_dict(weight)
            self.warm_epoch = -1

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        tag = False
        for i, (data, mask) in enumerate(tbar):

            data = data.to(self.device)
            labels = mask.to(self.device)

            if epoch > self.warm_epoch:
                tag = True

            masks, pred = self.model(data, tag)
            loss = 0

            loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)
            for j in range(len(masks)):
                if j>0:
                    labels = self.down(labels)
                loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch)

            loss = loss / (len(masks) + 1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))

        self.scheduler.step()
    def test(self, epoch):
        if args.save_img:

            dataset_dir = 'dataset/'+args.save_img_dataset
            train_img_ids, val_img_ids, test_txt = load_dataset('dataset', args.save_img_dataset)

            visulization_path = 'result' + '/' + 'visulization_result'
            visulization_fuse_path = 'result' + '/' + 'visulization_fuse'
            os.makedirs(visulization_path, exist_ok=True)
            os.makedirs(visulization_fuse_path, exist_ok=True)
            make_visulization_dir(visulization_path, visulization_fuse_path)

        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()


        tbar = tqdm(self.val_loader)
        tag = False
        with torch.no_grad():
            num = 0
            for i, (data, mask) in enumerate(tbar):

                data = data.to(self.device)
                mask = mask.to(self.device)

                if epoch > self.warm_epoch:
                    tag = True

                loss = 0
                _, pred = self.model(data, tag)

                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()


                if args.save_img:
                    save_Pred_GT(pred, mask, visulization_path, val_img_ids, num, '.png')
                    num += 1

                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, mean_IoU))
            if args.save_img:
                total_visulization_generation(dataset_dir, test_txt, '.png', visulization_path, visulization_fuse_path)

            FA, PD = self.PD_FA.get(len(self.val_loader))
            IoU, mean_IoU = self.mIoU.get()
            ture_positive_rate, false_positive_rate, _, _ ,F1= self.ROC.get()

            if self.mode == 'train':
                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU

                    torch.save(self.model.state_dict(), self.save_folder + '/weight.pkl')
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'.
                                format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                       epoch, self.best_iou*100, PD[0]*100, FA[0] * 1000000))

                all_states = {"net": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "epoch": epoch,
                              "iou": self.best_iou}
                torch.save(all_states, self.save_folder + '/checkpoint.pkl')
            elif self.mode == 'test':
                print('mIoU:' + str(mean_IoU*100) + '\n')
                print('Pd: ' + str(PD[0]*100) + '\n')
                print('Fa: ' + str(FA[0] * 1000000) + '\n')
                print('F1: ' + str(F1 * 100) + '\n')



         
if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)
    if trainer.mode=='train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)
    else:
        start_time=time.time()
        trainer.test(1)
        end_time=time.time()
        print('Time elapsed: ' + str(end_time-start_time))