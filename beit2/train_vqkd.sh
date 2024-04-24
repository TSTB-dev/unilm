    # parser.add_argument('--batch_size', default=64, type=int)
    # parser.add_argument('--epochs', default=100, type=int)
    # parser.add_argument('--save_ckpt_freq', default=20, type=int)
    # # Model parameters
    # parser.add_argument('--model', default='vqkd_encoder_base_decoder_3x768x12_clip', type=str, metavar='MODEL',  help='Name of model to train')  

    # parser.add_argument('--rec_loss_type', default='cosine', type=str, metavar='MODEL',
    #                     help='type of loss to calculate reconstruction distance')

    # parser.add_argument('--codebook_n_emd', default=8192, type=int, metavar='MODEL',
    #                     help='number of codebook')
    # parser.add_argument('--codebook_emd_dim', default=32, type=int, metavar='MODEL',
    #                     help='number of codebook')
    # parser.add_argument('--ema_decay', default=0.99, type=float, metavar='MODEL', help='ema decay for quantizer')
    # parser.add_argument('--quantize_kmeans_init', action='store_true', help='enable kmeans_init for quantizer')

    # parser.add_argument('--process_type', default='default', type=str, choices=['default', 'dall-e', 'imagenet_norm'],
    #                     help='Image process type (default, dall-e)')
    # parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')

    # # regress feature
    # parser.add_argument('--teacher_model_type', default='clip', type=str, help='teacher_model_type during training')
    # parser.add_argument('--teacher_input_size', default=224, type=int, help='teacher_input_size for clip-large p14')

    # # Optimizer parameters
    # parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
    #                     help='Optimizer (default: "adamw"')
    # parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
    #                     help='Optimizer Epsilon (default: 1e-8)')
    # parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
    #                     help='Optimizer Betas (default: None, use opt default)')
    # parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
    #                     help='Clip gradient norm (default: None, no clipping)')
    # parser.add_argument('--weight_decay', type=float, default=1e-4,
    #                     help='weight decay (default: 1e-4)')
    # parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
    #     weight decay. We use a cosine schedule for WD. 
    #     (Set the same value with args.weight_decay to keep weight decay no change)""")

    # parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
    #                     help='learning rate (default: 5e-5)')
    # parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
    #                     help='warmup learning rate (default: 1e-6)')
    # parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    # parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
    #                     help='epochs to warmup LR, if scheduler supports')
    # parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
    #                     help='epochs to warmup LR, if scheduler supports')

    # # Augmentation parameters
    # parser.add_argument('--color_jitter', type=float, default=0., metavar='PCT',
    #                     help='Color jitter factor (default: 0.)')
    # parser.add_argument('--train_interpolation', type=str, default='bicubic',
    #                     help='Training interpolation (random, bilinear, bicubic, lanczos default: "bicubic")')
    # parser.add_argument('--min_crop_scale', type=float, default=0.08, metavar='PCT',
    #                     help='min_crop_scale (default: 0.08)')

    # # Dataset parameters
    # parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
    #                     help='dataset path')
    # parser.add_argument('--eval_data_path', default='', type=str, help='dataset path')
    # parser.add_argument('--data_set', default='image_folder', type=str, help='dataset path')
 
    # parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    # parser.add_argument('--output_dir', default='',
    #                     help='path where to save, empty for no saving')
    # parser.add_argument('--log_dir', default=None,
    #                     help='path where to tensorboard log')
    
    # parser.add_argument('--device', default='cuda',
    #                     help='device to use for training / testing')
    # parser.add_argument('--seed', default=0, type=int)
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    # parser.add_argument('--auto_resume', action='store_true')
    # parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    # parser.set_defaults(auto_resume=True)

    # parser.add_argument('--dist_eval', action='store_true', default=True,
    #                     help='Enabling distributed evaluation')
    # parser.add_argument('--disable_eval', action='store_true', default=False)
    
    # parser.add_argument('--eval', action='store_true', default=False, help="Perform evaluation only")
    # parser.add_argument('--calculate_codebook_usage', action='store_true', default=False)

    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    # parser.add_argument('--num_workers', default=10, type=int)
    # parser.add_argument('--pin_mem', action='store_true',
    #                     help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    # parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
    #                     help='')
    # parser.set_defaults(pin_mem=True)
    
    # # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--local_rank', default=-1, type=int)
    # parser.add_argument('--dist_on_itp', action='store_true')
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

/home/sakai/projects/Reimpl/BEiTv2/beit2/bin/python3 /home/sakai/projects/Reimpl/BEiTv2/unilm/beit2/run_vqkd_training.py \
    --batch_size 32 \
    --epochs 500 \
    --save_ckpt_freq 100 \
    --model vqkd_encoder_base_decoder_1x768x12_dino \
    --rec_loss_type cosine \
    --codebook_n_emd 20 \
    --codebook_emd_dim 32 \
    --ema_decay 0.99 \
    --quantize_kmeans_init \
    --process_type default \
    --input_size 384 \
    --teacher_model_type dino \
    --teacher_input_size 384 \
    --opt adamw \
    --opt_eps 1e-8 \
    --opt_betas 0.9 0.999 \
    --clip_grad 1.0 \
    --weight_decay 1e-4 \
    --weight_decay_end 1e-4 \
    --lr 5e-5 \
    --warmup_lr 1e-6 \
    --min_lr 1e-5 \
    --warmup_epochs 5 \
    --warmup_steps -1 \
    --color_jitter 0.0 \
    --train_interpolation bicubic \
    --min_crop_scale 0.00 \
    --data_root /home/sakai/projects/LADMIM/LADMIM/data/mvtec_loco \
    --category box \
    --eval_data_path /home/sakai/projects/LADMIM/LADMIM/data/mvtec_loco \
    --data_set loco \
    --imagenet_default_mean_and_std \
    --output_dir /home/sakai/projects/Reimpl/BEiTv2/unilm/beit2/output/vqkd_encoder_base_decoder_1x768x12_dino \
    --log_dir /home/sakai/projects/Reimpl/BEiTv2/unilm/beit2/output/vqkd_encoder_base_decoder_1x768x12_dino \
    --device cuda \
    --seed 0 \


