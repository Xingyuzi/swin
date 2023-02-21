from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
import cv2
import numpy as np
import mmcv

def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--save_folder', default='./widerface_evaluate/swin_tiny3x_v2/', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='/home/techart/xyz/xyz/xyz/data/WIDER_FACE/images/', type=str, help='dataset path')
    # parser.add_argument(
    #     '--img',
    #     type=str,
    #     # default='/home/techart/xyz/swin/swin_master/data/WIDERFace/WIDER_train/0--Parade/0_Parade_marchingband_1_849.jpg', # XYZ TODO
    #     default='/home/techart/xyz/swin/swin_master/data/WIDERFace/WIDER_train/0--Parade/0_Parade_Parade_0_904.jpg', # XYZ TODO
    #     help='Input image path') # XYZTODO
    # parser.add_argument('--config',default="/home/techart/xyz/swin/swin_master/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py", 
    #     help='test config file path') # XYZTODO
    # # parser.add_argument('--config',default="/home/techart/xyz/swin/swin_master/configs/swin/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py", 
    # #     help='test config file path') # XYZTODO
    # parser.add_argument('--checkpoint',default="/home/techart/xyz/swin/swin_master/work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco_3x/latest.pth", 
    #     help='checkpoint file') # XYZTODO
    parser.add_argument('--config',default="/home/techart/xyz/swin/swin_master/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco_v2.py", 
        help='test config file path') # XYZTODO
    parser.add_argument('--checkpoint',default="/home/techart/xyz/swin/swin_master/work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco_v2/latest.pth", 
        help='checkpoint file') # XYZTODO
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    score_thr = args.score_thr
    testset_folder = args.dataset_folder
    testset_list = "/home/techart/xyz/swin/swin_master/data/WIDERFace/val.txt"
    with open(testset_list, 'r') as fr:
        # test_dataset = [x.strip() for x in fr.readlines()]
        test_dataset = []
        for line in fr.readlines():
            key = "jpg"
            if key in line:
                test_dataset.append(line.strip())

    

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)  
    
    num_images = len(test_dataset)  
    for i, img_name in enumerate(test_dataset):
       
        image_path = testset_folder + img_name  # TODO WIDER_FACE
        assert os.path.exists(image_path), print("there is no filepath", image_path)
        # img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        result = inference_detector(model, image_path)
        
        # XYZ copy from base.py
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        # XYZ copy from /home/techart/xyz/swin/swin_master/mmdet/core/visualization/image.py
        assert bboxes.ndim == 2, \
            f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
        assert labels.ndim == 1, \
            f' labels ndim should be 1, but its ndim is {labels.ndim}.'
        assert bboxes.shape[0] == labels.shape[0], \
            'bboxes.shape[0] and labels.shape[0] should have the same length.'
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
            f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
        # img = mmcv.imread(img).astype(np.uint8)

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
        
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            # bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            fd.write(file_name)
            # fw.write('{:.1f}\n'.format(dets.shape[0]))
            bboxes_num = str(len(bboxes)) + "\n"
            fd.write(bboxes_num)

            for i,box in enumerate(bboxes):
                # if box[-1] == 1:
                 x = int(box[0])
                 y = int(box[1])
                 w = int(box[2]) - int(box[0])
                 h = int(box[3]) - int(box[1])
                 confidence = str(box[-1])
                 line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                 fd.write(line)
        


    
    
    # test a single image
    # result = inference_detector(model, args.img)
    # print("##############\n",result)
    # show the results
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
