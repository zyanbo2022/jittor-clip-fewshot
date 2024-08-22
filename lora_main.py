
import clip
from datasets.utils import DatasetWrapper
from utils_lora import *
from run_utils import *
from lora import run_lora
from datasets.TrainSet import TrainSet_double

def main():

    # Load config file
    args = get_arguments()
    
    set_random_seed(args.seed)
    
    # CLIP
    clip_model, preprocess = clip.load('ViT-B-32.pkl')
    clip_model.eval()
    logit_scale = 100

    # Prepare dataset
    print("Preparing dataset.")
    dataset =TrainSet_double()
    val_loader = DatasetWrapper(data_source=dataset.val, batch_size=80, is_train=False, tfm=tfm_test_base, shuffle=False)
    test_loader = DatasetWrapper(data_source=dataset.test, batch_size=80, is_train=False, tfm=tfm_test_base, shuffle=False)

        
    train_loader = None
    if not args.eval_only:

        train_loader = DatasetWrapper(data_source=dataset.train_x, batch_size=32, tfm=tfm_train_base, is_train=True, shuffle=True)
        
    args_dict = vars(args)
    message = '\n'.join([f'{k:<20}: {v}' for k, v in args_dict.items()])
    print(message)
    
    run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)

if __name__ == '__main__':
    main()



