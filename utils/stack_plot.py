import matplotlib.pyplot as plt
import os


def stack_swapped_plot(input_img_dir,
                       H_crop_factor=0.2,
                       W_crop_factor=0.55,
                       H_shift=15,
                       W_shift=40):
    images = sorted(os.listdir(input_img_dir))
    gt_list = list()
    recon_list = list()
    swapped_list = list()
    for img in images:
        if 'gt' in img:
            gt_list.append(os.path.join(input_img_dir, img))
        elif 'recon' in img:
            recon_list.append(os.path.join(input_img_dir, img))
        else:
            swapped_list.append(os.path.join(input_img_dir, img))

    # every image has H=240, W=320 pixels
    # crop every single images to get a bette view
    # because it may have too much empty place around the full shape
    H_img_start = int(240*H_crop_factor/4) + H_shift
    W_img_start = int(320*W_crop_factor/4) + W_shift
    H_img_end = int(H_img_start+240*(1-H_crop_factor))
    W_img_end = int(W_img_start+320*(1-W_crop_factor))
    H_fig = 2.4*(1-H_crop_factor)*2  # 2 images per column
    W_fig = 3.2*(1-W_crop_factor)*6  # 6 images per row

    fig = plt.figure(figsize=(W_fig, H_fig))
    for i in range(2):
        for j in range(6):
            ax = fig.add_axes([j/6, (1-i)/2, 1/6, 1/2])
            ax.axis('off')
            if j == 0:
                ax.imshow(plt.imread(gt_list[2*i])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 1:
                ax.imshow(plt.imread(recon_list[2*i])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 2:
                ax.imshow(plt.imread(swapped_list[2*i])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 3:
                ax.imshow(plt.imread(swapped_list[2*i+1])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 4:
                ax.imshow(plt.imread(recon_list[2*i+1])[H_img_start:H_img_end, W_img_start:W_img_end])
            else:
                ax.imshow(plt.imread(gt_list[2*i+1])[H_img_start:H_img_end, W_img_start:W_img_end])

    plt.savefig(os.path.join(input_img_dir, 'stacked_image.png'))


def stack_interpolation_plot(input_img_dir,
                             H_crop_factor=0.2,
                             W_crop_factor=0.55,
                             H_shift=15,
                             W_shift=40):
    images = sorted(os.listdir(input_img_dir))
    gt_list = list()
    recon_list = list()
    shape_list = list()
    part_list = list()
    for img in images:
        if 'gt' in img:
            gt_list.append(os.path.join(input_img_dir, img))
        elif 'recon' in img:
            recon_list.append(os.path.join(input_img_dir, img))
        elif 'shape' in img:
            shape_list.append(os.path.join(input_img_dir, img))
        else:
            part_list.append(os.path.join(input_img_dir, img))

    # every image has H=240, W=320 pixels
    # crop every single images to get a bette view
    # because it may have too much empty place around the full shape
    H_img_start = int(240*H_crop_factor/4) + H_shift
    W_img_start = int(320*W_crop_factor/4) + W_shift
    H_img_end = int(H_img_start+240*(1-H_crop_factor))
    W_img_end = int(W_img_start+320*(1-W_crop_factor))
    H_fig = 2.4*(1-H_crop_factor)*2  # 2 images per column
    W_fig = 3.2*(1-W_crop_factor)*12  # 12 images per row

    fig = plt.figure(figsize=(W_fig, H_fig))
    for i in range(2):
        for j in range(12):
            ax = fig.add_axes([j/12, (1-i)/2, 1/12, 1/2])
            ax.axis('off')
            if j == 0:
                ax.imshow(plt.imread(gt_list[0])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 1:
                ax.imshow(plt.imread(recon_list[0])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 10:
                ax.imshow(plt.imread(recon_list[1])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 11:
                ax.imshow(plt.imread(gt_list[1])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j != 0 and j != 1 and j != 10 and j != 11 and i == 0:
                ax.imshow(plt.imread(shape_list[j-2])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j != 0 and j != 1 and j != 10 and j != 11 and i == 1:
                ax.imshow(plt.imread(part_list[j-2])[H_img_start:H_img_end, W_img_start:W_img_end])

    plt.savefig(os.path.join(input_img_dir, 'stacked_image.png'))


def stack_assembly_plot():
    pass
