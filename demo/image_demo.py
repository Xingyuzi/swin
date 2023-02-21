from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--img',
        type=str,
        # default='/home/techart/xyz/swin/swin_master/data/WIDERFace/WIDER_train/0--Parade/0_Parade_marchingband_1_849.jpg', # XYZ TODO
        default='/home/techart/xyz/swin/swin_master/demo/0_Parade_marchingband_1_849.jpg', # XYZ TODO
        help='Input image path') # XYZTODO
    # parser.add_argument('--config',default="/home/techart/xyz/swin/swin_master/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py", 
    #     help='test config file path') # XYZTODO
    parser.add_argument('--config',default="/home/techart/xyz/swin/swin_master/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco_cosattn.py", 
        help='test config file path') # XYZTODO
    parser.add_argument('--checkpoint',default="/home/techart/xyz/swin/swin_master/work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco_cosattn/latest.pth", 
        help='checkpoint file') # XYZTODO
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    print("##############\n",result)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
