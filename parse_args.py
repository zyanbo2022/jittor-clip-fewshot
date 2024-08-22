
import argparse
import yaml

def parse_args():
    # default config
    cfg = yaml.load(open('default.yaml', 'r'), Loader=yaml.Loader)
    
    parser = argparse.ArgumentParser(description='CLIP Few Shot')

    parser.add_argument('--dataset',
                        type=str,
                        default=cfg['dataset'],
                        help='name of dataset')
    
    parser.add_argument('--shots',
                        type=int,
                        default=cfg['shots'])
                        
    parser.add_argument('--root_path',
                        type=str,
                        default=cfg['root_path'],
                        help='root path of dataset') 
    
    parser.add_argument('--rand_seed',
                        type=int,
                        default=cfg['rand_seed'],
                        help='rand_seed') 
    
    parser.add_argument('--train_epoch',
                        type=int,
                        default=cfg['train_epoch'],
                        help='train epochs')
    
    parser.add_argument('--lr',
                        type=float, default=cfg['lr'], metavar='LR',
                        help='learning rate')
    
    parser.add_argument('--load_pre_feat', 
                        default=cfg['load_pre_feat'],
                        help='load test features or not')


    parser.add_argument('--load_aux_weight',
                        default=cfg['load_aux_weight'],
                        help='load aux features weight')
    
    parser.add_argument('--load_cache',
                        default=cfg['load_cache'],
                        help='load cache features weight')
    
    parser.add_argument('--clip_backbone',
                        type=str,
                        default=cfg['clip_backbone'],
                        help='name of clip backbone')
    

    parser.add_argument('--num_classes',
                        type=int,
                        default=cfg['num_classes'],
                        help='model classification num') 
    
    parser.add_argument('--augment_epoch',
                    type=int,
                    default=cfg['augment_epoch'],
                    help='augment epoch')
    
    parser.add_argument('--alpha',
                    type=float,
                    default=cfg['alpha'],
                    help='alpha')
    
    parser.add_argument('--alphatip',
                    type=float,
                    default=cfg['alphatip'],
                    help='tip alpha')
    
    parser.add_argument('--lambda_merge',
                    type=float,
                    default=cfg['lambda_merge'],
                    help='merge loss ratio'
                    )
    parser.add_argument('--uncent_type',
                    type=str,
                    default='moment',
                    help='uncertainty fusion'
                    )
    parser.add_argument('--uncent_power',
                    type=float,
                    default=0.4,
                    help='uncertainty fusion power'
                    )
    return parser