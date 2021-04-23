import argparse

def ArgsParser():
        parser = argparse.ArgumentParser()
        # Datasets
        parser.add_argument('--dataset_root', type=str, default="/media/sda/datasets/IHD", help='frequency of showing training results on screen')
        parser.add_argument('--name', type=str, default='', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=224, help='then crop to this size')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        parser.add_argument('--mean', type=str, default='0.485, 0.456, 0.406', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--std', type=str, default='0.229, 0.224, 0.225', help='which epoch to load? set to latest to use latest cached model')
        
        # Display
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
        
        # network saving and loading parameters
        parser.add_argument('--input_nc', type=int, default=3, help='# of iter at starting learning rate')
        parser.add_argument('--output_nc', type=int, default=1, help='# of iter at starting learning rate')

        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        
        # training parameters
        parser.add_argument('--resume', type=int, default=-3, help='# of iter at starting learning rate')
        parser.add_argument('--nepochs', type=int, default=60, help='# of iter at starting learning rate')
        parser.add_argument('--pretrain_nepochs', type=int, default=50, help='# of iter at starting learning rate')
        
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='# weight decay for the optimizer')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        parser.add_argument('--sync_bn', action='store_true', help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lambda_attention', type=float, default=1, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lambda_detection', type=float, default=1, help='multiply by a gamma every lr_decay_iters iterations')

        parser.add_argument('--is_train', type=int, default=1, help='# of iter at starting learning rate')
        parser.add_argument('--port', type=str, default='tcp://192.168.1.201:12345', help='# of iter at starting learning rate')
        parser.add_argument("--local_rank", type=int, default=0)

        parser.add_argument('--backbone', type=str, default='resnet34', help='# of iter at starting learning rate')
        parser.add_argument('--ggd_ch', type=int, default=32, help='# of iter at starting learning rate')

        parser.add_argument('--mda_mode', type=str, default='vanilla', help='# of iter at starting learning rate')
        parser.add_argument('--loss_mode', type=str, default='', help='# of iter at starting learning rate')
        parser = parser.parse_args()
        return parser