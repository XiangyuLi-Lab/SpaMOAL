import argparse
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import time
import numpy as np
import pandas as pd
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,10000).__str__()

def position_info(filename):
    return pd.read_csv(filename, usecols=[1, 3, 4], names=['a', 'b', 'c']).values.tolist()

def get_partition(img, rows, cols, width, height, size):
    # if size == 224:
    #     a = b = 112
    # else:
    #     a = 149
    #     b = 150
    a = b = size/2
    # 计算边界
    x_left = width - a
    y_left = height - b
    x_right = width + b
    y_right = height + a

    # 调整边界以适应图像尺寸
    x_left = int(max(0, x_left))
    y_left = int(max(0, y_left))
    x_right = int(min(cols, x_right))
    y_right = int(min(rows, y_right))

    return img[y_left:y_right, x_left:x_right]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="E11_0")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    # args.cuda = torch.cuda.is_available()
    print(args)
    list = []
    position = pd.read_csv(f"../Data/{args.name}/{args.name}_position.csv",  index_col=False, names=['a', 'b', 'c']).values.tolist()
    position = position[1:]
    print(position)
    start = time.time()
    size = args.size
    img = cv2.imread(f"../Data/{args.name}/{args.name}_HE.png")
    print(img)
    rows, cols = int(img.shape[0]), int(img.shape[1])
    os.makedirs(f'../Data/{args.name}/img_{args.name}_{args.size}', exist_ok=True)
    print("mkdir sucessful!")
    # 统计第二列和第三列的最大值
    max_x = int(float(max([item[1] for item in position])))
    max_y = int(float(max([item[2] for item in position])))
    x_scale = cols / max_x
    y_scale = rows / max_y
    for i in range(len(position)):
        x = int(round(float(position[i][1])))
        y = int(round(float(position[i][2])))
        img_part = get_partition(img,rows,cols, x, y, size)
        cut_path = 'img/'+args.name+'/' +str(args.size)+'/' + position[i][0]+'.png'  # .jpg图片的保存路径
        cv2.imwrite(cut_path, img_part)

        img_part_npy = np.array(img_part)
        cut_path_npy = f'../Data/{args.name}/img_{args.name}_{args.size}' + '/' + position[i][0]+ '.npy'  
        #cut_path_npy = f'./test_time/img_{args.name}_{args.size}' + '/' + position[i][0] + '.npy'
        np.save(cut_path_npy,img_part_npy)
        # print(position[i][0]," was saved!!")
    
    end = time.time()
    print("run time: "+ str(end-start) + " seconds")




