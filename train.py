import argparse
import os.path
import numpy as np
import torch
import timm
import torchvision.transforms
import tqdm
import clip
import glob
from PIL import Image
from matplotlib import pyplot as plt


def get_arg_parser():
    parser = argparse.ArgumentParser('Image search task', add_help=False)
    # model parameter
    parser.add_argument('--input_size', default=128, type=int)

    parser.add_argument('--datasets_dir', default='/home/tonnn/.nas/weijia/datasets/fruit-and-vegetable-image-recognition/train\
', help='path where the train sets loadin')

    parser.add_argument('--test_image_dir', default='/home/tonnn/.nas/weijia/datasets/fruit-and-vegetable-image-recognition/test\
')
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--model_name', default='clip')
    parser.add_argument('--feature_dict_file', default='corpus_feature_file_dict.npy')
    parser.add_argument('--topk', default=7, type=int)
    parser.add_argument('--mode', default='search')

    return parser


def extract_feature_by_CLIP(model, preprocess, img):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = preprocess(Image.open(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(img)
        vec = vec.squeeze().cpu().numpy()
    return vec


def extract_feature_single(args, model, img):
    """
    这个使用热水net的需要进行图片预处理才可以得到结果
    和之前一样，转化为rgb，然后进行关于i异化，方差
    :param args:
    :param model:
    :param img:
    :return:
    """
    img_rgb = Image.open(img).convert('RGB')
    image = img_rgb.resize((args.input_size, args.input_size), Image.ANTIALIAS)
    image = torchvision.transforms.ToTensor()(image)

    trainset_mean = [0.47083899, 0.43284143, 0.3242959]
    trainset_std = [0.37737389, 0.36130483, 0.34895992]

    image = torchvision.transforms.Normalize(mean=trainset_mean, std=trainset_std)(image).unsqueeze(0)

    with torch.no_grad():
        feature = model.forward_features(image)
        vec = model.global_pool(feature)
        vec = vec.squeeze().numpy()

    img_rgb.close()
    return vec


def extract_feature(args, model, image_path='', preprocess=None):
    """
    提取图像特征，放入到字典进行储存
    clip直接调用，resnet，需要使用初始化，然后把最后的向量的向量放进去
    :return:
    """
    allVectors = {}
    for img in tqdm.tqdm(glob.glob(os.path.join(image_path, '*', '*.jpg'))):
        if args.model_name == 'clip':
            allVectors[img] = extract_feature_by_CLIP(model, preprocess, img)
        else:
            allVectors[img] = extract_feature_single(args, model, img)
    # 得到之后就进行存放
    os.makedirs(f"{args.save_dir}/{args.model_name}", exist_ok=True)
    np.save(f"{args.save_dir}/{args.model_name}/{args.feature_dict_file}", allVectors)
    return allVectors


def getSimilarityMatrix(Vectors_dict):
    v = np.array(list(Vectors_dict.values()))  # [num,h]
    numerator = np.matmul(v, v.T)
    denominator = np.matmul(np.linalg.norm(v, axis=1, keepdims=True),
                            np.linalg.norm(v, axis=1, keepdims=True).T)  # num,num]
    sim = numerator / denominator
    keys = list(Vectors_dict.keys())
    return sim, keys


def setAxes(ax, image, query=False, **kwargs):
    value = kwargs.get("value", None)
    if query:
        ax.set_xlabel("Query IMage\n".format(image), fontsize=12)
        ax.xaxis.label.set_color('red')
    else:
        ax.set_xlabel("score={1:1.3f}\n{0}".format(image, value), fontsize=12)
        ax.xaxis.label.set_color('blue')
    ax.set_xticks([])
    ax.set_yticks([])


def plotSimilarIMages(args, img, simImages, simValues, numRow=1, numCol=4):
    fig = plt.figure()

    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(f"use engine model:{args.model_name}", fontsize=35)

    for j in range(0, numRow * numCol):
        ax = []
        if j == 0:
            # 询问的图像
            image = Image.open(img)
            ax = fig.add_subplot(numRow, numCol, 1)
            setAxes(ax, img.split(os.sep)[-1], query=True)
        else:
            # 相似的图像
            image = Image.open((simImages[j - 1]))
            ax.append(fig.add_subplot(numRow, numCol, j + 1))
            setAxes(ax[-1], simImages[j - 1].split(os.sep)[-1], value=simValues[j - 1])
        image = image.convert('RGB')
        plt.imshow(image)
        image.close()
    fig.savefig(f"{args.save_dir}/{args.model_name}_search_top_{args.topk}_{img.split(os.sep)[-1].split('.')[0]}.png")
    plt.show()


def main():
    args = get_arg_parser().parse_args()

    if args.model_name != 'clip':
        model = timm.create_model(args.model_name, pretrained=True)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of parameters: %.2f' % (n_parameters / 1.e6))
        preprocess = None
        # 创建模型，然后计算参数量，最后进行放入到测试模式
        model.eval()
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("finish making clip")

    if args.mode == 'extract':
        print(args.datasets_dir)
        allVector = extract_feature(args, model,image_path=args.datasets_dir,preprocess=preprocess)
        print('finish extract')
    else:
        print(f"use pretrain model {args.model_name} to search {args.topk} similar image from corpus")

        test_images = glob.glob(os.path.join(args.test_image_dir,  '*', '*.jpg'))
        test_images += glob.glob(os.path.join(args.test_image_dir, '*', '*.jpeg'))
        test_images += glob.glob(os.path.join(args.test_image_dir, '*', '*.png'))

        # load vector dictionary
        print("ok")
        allVector = np.load(f"{args.save_dir}/{args.model_name}/{args.feature_dict_file}", allow_pickle=True)

        allVector = allVector.item()

        # 加入读取的测试图像，也放入到corpus
        for img in tqdm.tqdm(test_images):
            print(f"reading {img}...")

            if args.model_name == 'clip':
                allVector[img] = extract_feature_by_CLIP(model, preprocess, img)
            else:
                allVector[img] = extract_feature_single(args, model, img)

        sim, keys = getSimilarityMatrix(allVector)
        # 倒叙进行取值
        result = {}

        for img in tqdm.tqdm(test_images):
            print(f"sorting most similarity images as {img}...")
            index = keys.index(img)
            sim_vec = sim[index]
            # sort然后逆序，然后取值
            indexs = np.argsort(sim_vec)[::-1][1:args.topk]
            simImages = []
            simScores = []
            for ind in indexs:
                simImages.append(keys[ind])
                simScores.append(sim_vec[ind])
            result[img] = (simImages, simScores)

        print(f"showing the pictures")
        for img in tqdm.tqdm(test_images):
            plotSimilarIMages(args, img, result[img][0], result[img][1], numRow=1, numCol=args.topk)


if __name__ == '__main__':
    print("hi")
    main()
