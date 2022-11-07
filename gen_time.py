# 生成图
import json
import os
import random
import sys

# from imgcat import imgcat
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm


# from tqdm._tqdm import trange

def get_list(path, type):
    list_out = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(type)])
    return list_out


def makesure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def getUints(units_type):
    if units_type == 'time':
        return ['年', '月', '日', '时', '分']
    if units_type == 'time':
        return ['小时', '分钟', '分', '秒', '年', '月', '日', '刻', '半小时', '毫秒']
    if units_type == 'long':
        return ['厘米', '毫米', '分米', '米', '公里', 'cm', 'm', 'dm']
    if units_type == 'weight':
        return ['千克', '克', '公斤', '斤', '吨', 'kg', 't']
    if units_type == 'area':
        return ['平方米', '平方分米', '平方厘米', '平方公里', '公顷']
    if units_type == 'money':
        return ['元', '角', '分']
    if units_type == 'wan':
        return ['万', '亿']
    if units_type == 'volume':
        return ['立方米', '立方分米', '立方厘米', '毫升', '升']


def random_num():
    num1 = np.random.randint(0, 100)
    num2 = np.random.randint(0, 10)
    return num1, num2


def random_unit():
    # units_type = random.choice(['time','long','weight','area','money',\
    #     'wan','volume'])
    units_type = "time"
    uint_list = getUints(units_type)
    unit1 = np.random.choice(uint_list, random.randint(3, 5), p=[0.3, 0.3, 0.3, 0.05, 0.05])

    return unit1


def filter_op(op):
    threshold = 225
    op_data = np.array(op)
    # imgcat(op_data)
    mask = np.where(op_data > threshold)
    nomask = np.where(op_data <= threshold)
    op_data[mask] = 255
    k = 255.0 / threshold
    op_data[nomask] = op_data[nomask] * k
    op_data = np.uint8(op_data)

    return op_data


def pre_paste(path):
    num_data = Image.open(path).convert("RGB")
    num_data = filter_op(num_data)

    return num_data


def resize_num3(num_datas):
    out = []
    for num_data in num_datas:
        op_h, op_w = num_data.shape[0], num_data.shape[1]
        new_w = int(64 * op_w / op_h)
        op_data = cv2.resize(num_data, (new_w, 64), interpolation=cv2.INTER_AREA)
        out.append(op_data)
    return out


def resize_paste(bg_data, num_data, end):
    bg = bg_data.copy()
    bg_h, bg_w = bg_data.shape[0], bg_data.shape[1]
    # end = bg_w-1

    op_h, op_w = num_data.shape[0], num_data.shape[1]
    new_w = int(64 * op_w / op_h)
    op_data = cv2.resize(num_data, (new_w, 64), interpolation=cv2.INTER_AREA)

    bg_crop = bg[:, end:end + new_w, :]

    bg_crop = bg_crop / 255.
    op_data = op_data / 255.

    mix = bg_crop * op_data
    mix = np.round((mix * 255.)).astype(np.uint8)
    bg[:, end:end + new_w, :] = mix
    end = end + new_w
    # imgcat(bg)
    # sys.exit()
    return end, bg


def paste_num_v2(path, bg_data, start, thres=128):
    num_data = cv2.imread(path)
    bg = bg_data.copy()
    data_h, data_w = num_data.shape[0], num_data.shape[1]
    new_w = int(32 * data_w / data_h)
    op_data = cv2.resize(num_data, (new_w, 32), interpolation=cv2.INTER_AREA)
    op_data = filter_op(op_data)
    h = op_data.shape[0]
    end = start + new_w
    bg[:h, start:end, :] = op_data
    return bg, end


def paste_num(bg, path_num, end):
    num_data = pre_paste(path_num)

    bg_data = np.array(bg)
    bg_h, bg_w = bg_data.shape[0], bg_data.shape[1]

    w_use, bg_data = resize_paste(bg_data, num_data, end)

    return w_use, Image.fromarray(bg_data)


def outbbox(t_size, part1, part2, imshape, new_w, font):
    imh, imw = imshape

    fontText1 = ImageFont.truetype(font, size=t_size, encoding="utf-8")
    tw1, th1 = fontText1.getsize(part1)

    fontText2 = ImageFont.truetype(font, size=t_size, encoding="utf-8")
    tw2, th2 = fontText2.getsize(part2)

    total_w = new_w + tw1 + tw2

    if total_w > imw - 10 or th1 > imh - 2 or th2 > imh - 2:
        return True
    else:
        return False


def compute_tsize(part1, part2, path_num2, imshape):
    bgh, bgw = imshape
    font = './fonts/%02d.ttf' % (np.random.randint(0, 28))
    # 计算图片resize后的宽度
    num_data = cv2.imread(path_num2)
    op_h, op_w = num_data.shape[0], num_data.shape[1]
    new_w = int(bgh * op_w / op_h)

    t_size = np.random.randint(30, 60)
    # 计算
    while outbbox(t_size, part1, part2, imshape, new_w, font):
        t_size -= 1
    return t_size, font



def create(img_dir, year_list, month_list, day_list, hour_list, min_list):
    # bg = random.choice(bg_list)
    bg = "./datasets/bg/bg2.jpg"
    unit1 = random_unit()
    unit1 = list(set(unit1))
    rule = {"年": 0, "月": 1, "日": 2, "时": 3, "分": 4}
    unit1_list = sorted(unit1, key=lambda x: rule[x])
    # 打开背景图
    im_bg = Image.open(bg)
    bg_data = np.array(im_bg)

    color = (0, 0, 0)
    # 遍历年月日时分，分别从对应的文件夹选择图片
    tleft = 0

    font_dir = "./fonts"
    font = random.choice([os.path.join(font_dir, file) for file in os.listdir(font_dir) if file.endswith("ttf")])
    t_size = np.random.randint(28, 32)
    shift = random.randint(0, 5)
    ttop = shift
    content = ""

    p = random.random()
    replace_char = ""
    if 0.9<p<1:
        replace_char = "."
        font = "./fonts/品如手写体.ttf"


    fontText = ImageFont.truetype(font, t_size, encoding="utf-8")

    for c in unit1_list:
        if c == "年":
            if len(replace_char):
                c = replace_char
            line = random.choice(year_list)
            path = os.path.join(img_dir, line["filename"])
            content += line["label"]
        elif c == "月":
            if len(replace_char):
                c = replace_char
            line = random.choice(month_list)
            path = os.path.join(img_dir, line["filename"])
            content += line["label"]

        elif c == "日":
            if len(replace_char):
                c = replace_char
            line = random.choice(day_list)
            path = os.path.join(img_dir, line["filename"])
            content += line["label"]

        elif c == "时":
            if len(replace_char):
                c = replace_char
            line = random.choice(hour_list)
            path = os.path.join(img_dir, line["filename"])
            content += line["label"]

        else:
            if len(replace_char):
                c = replace_char
            line = random.choice(min_list)
            path = os.path.join(img_dir, line["filename"])
            content += line["label"]


        # 先贴手写字
        image = cv2.imread(path)
        h, w, _ = image.shape
        bg_data = np.array(bg_data)
        bg_data, end = paste_num_v2(path, bg_data, tleft)
        # 再贴印刷字
        tleft = end + shift
        bg_data = Image.fromarray(bg_data)
        draw = ImageDraw.Draw(bg_data)
        # 加上年月日的距离
        if random.random() > 0.9:
            continue
        draw.text((tleft, ttop), c, color, font=fontText)
        tw, th = fontText.getsize(c)

        tleft += tw
        content += c

    bg_data = np.array(bg_data)
    return Image.fromarray(bg_data[:, :tleft + 5, :]), content


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    # 图片路径

    # 输出路径
    out_dir = './output/syh_handwriting_002'
    out_img_dir = os.path.join(out_dir, 'images')
    makesure_path(out_dir)
    makesure_path(out_img_dir)
    label_path = os.path.join(out_dir, 'labels.txt')
    out_file = open(label_path, 'w')

    NUM = int(sys.argv[1])

    # 生成20w数据用于模型训练
    # NUM = 200

    year_pattern = set([str(i) for i in range(1000, 3000)])
    mon_pattern = set([str(i) for i in range(1, 13)] + [f"{i:02d}" for i in range(1, 10)])
    day_pattern = set([str(i) for i in range(1, 31)] + [f"{i:02d}" for i in range(1, 10)])
    hour_pattern = set([str(i) for i in range(1, 12)] + [f"{i:02d}" for i in range(0, 24)])
    min_pattern = set([str(i) for i in range(1, 10)] + [f"{i:02d}" for i in range(0, 60)])

    with open("datasets/label.json", encoding="utf-8") as rfile:
        labels = json.load(rfile)

    img_dir = "/data/old_data/wufan/database/20221009_手写日期_2500"
    # img_dir = "/Users/admin/Desktop/github/synth-data/datasets/img"
    year_list = [line for line in labels if line["label"] in year_pattern]
    month_list = [line for line in labels if line["label"] in mon_pattern]
    day_list = [line for line in labels if line["label"] in day_pattern]
    hour_list = [line for line in labels if line["label"] in hour_pattern]
    min_list = [line for line in labels if line["label"] in min_pattern]

    result = []
    for i in tqdm(range(NUM)):
        img, content = create(img_dir, year_list, month_list, day_list, hour_list, min_list)
        out_file.write(f'gen_image_{i:08d}.jpg\t{content}\n')
        result.append(f'gen_image_{i:08d}.jpg\t{content}\n')
        outname = os.path.join(out_img_dir, f'gen_image_{i:08d}.jpg')
        img.save(outname)

    out_file.close()

    random.seed(0)
    random.shuffle(result)
    x = int(len(result)*0.8)
    with open(os.path.join(out_dir,"train.txt"),'w',encoding="utf-8") as wfile:
        for line in result[: x]:
            wfile.writelines(line)

    with open(os.path.join(out_dir, "valid.txt"), 'w', encoding="utf-8") as wfile:
        for line in result[x:]:
            wfile.writelines(line)
    print('finish')
