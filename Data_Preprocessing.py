import os
import requests  # to get image from the web
import shutil
import pandas as pd
import time
from googletrans import Translator


class Data_Preprocessing:
    TRAIN_IMG_URL_IDX = 3
    TRAIN_IMG_ID_IDX = 4
    TRAIN_TITLE_IDX = 6
    TEST_IMG_URL_IDX = 0
    TEST_IMG_ID_IDX = 1

    def __init__(self):
        self.img_rt_folder = 'img'
        self.data_folder = r'C:\Users\yuxia\Documents\CS7311_Project\FIREWHEEL\data'

    def load_img(self, data_file, img_folder, img_url_idx, img_id_idx):
        """
        load_img download images from the url,
        save images into the given image folder
        and use image id as the image name
        :param data_file: input file which include information such as img_url, img_id
        :param img_folder: image folder where downloaded image are saved
        :param img_url_idx: column idx of img url in the data_file
        :param img_id_idx: column idx of img id in the data_file
        """
        f = open(os.path.join(self.data_folder, data_file), "r", encoding="utf-8")
        next(f)
        print("start loading images")
        for line in f:
            image_url = line.split("\t")[img_url_idx]
            image_id = line.split("\t")[img_id_idx]
            img_path = os.path.join(self.img_rt_folder, img_folder)
            isExist = os.path.exists(img_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(img_path)
                print("The image directory is created!")
            filename = os.path.join(img_path, image_id + ".jpg")
            r = requests.get(image_url, stream=True, headers={'User-agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                with open(filename, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
            else:
                print("img can't be loaded")

    def reformat_data_file(self, input_file, new_file):
        """
        reformat_data_file reformat the given input_file to facilate the further data processing
        :param input_file: origin tsv file
        :param new_file: output tsv file
        """
        f = open(os.path.join(self.data_folder, input_file), "r", encoding="utf-8")
        next(f)
        with open(new_file, 'a', encoding="utf-8") as the_file:
            header = "img_id, title"
            the_file.write(header + "\n")
            for line in f:
                image_id = line.split("\t")[self.TRAIN_IMG_ID_IDX] + ".jpg"
                title = line.split("\t")[self.TRAIN_TITLE_IDX]
                the_file.write(image_id + "\t" + title + "\n")

    def combine_files(self, filenames, output_file):
        """
        combine_csv combine a list of files into one file
        :param filenames: a list of filename
        :param output_file: output file
        """
        with open(output_file, "w", encoding="utf-8") as output:
            for i in range(len(filenames)):
                with open(filenames[i], "r", encoding="utf-8") as infile:
                    if i != 0:
                        next(infile)
                    contents = infile.read()
                    output.write(contents)

    def text_trans(self, file_path, title_idx):
        """

        :param file_path: original file including german article title
        :return: list of translated article title (in English)
        """
        file_text = open(file_path, 'r')
        translator = Translator()
        lines_text = file_text.readlines()
        cnt = 0
        trans_lines_text = []
        for l_text in lines_text:
            spes_text = l_text.split("\t")
            result_text = translator.translate(spes_text[title_idx], src='de')
            trans_lines_text.append(result_text.text)
            time.sleep(1)
            cnt += 1
            print(cnt)
            if cnt % 50 == 0:
                print("finish sub_lines_test_text ", cnt)
        return trans_lines_text

    def comb_title_eng(self, orig_file, titles_eng, output_file):
        """

        :param orig_file: original file including german article title
        :param titles_eng: list of translated article title (in English)
        :param output_file: output file including translated article title (in English)
        """
        lines = [line.strip() for line in open(orig_file, 'r', encoding="utf-8")]
        with open(output_file, 'a', encoding="utf-8") as the_file:
            for i in range(len(lines)):
                title_eng = titles_eng[i].rstrip("\n")
                segs = lines[i].strip("\n").split(",")
                the_file.write(segs[0] + "\t" + segs[1] + '\t' + title_eng + "\n")

    def trans_title(self, orig_file, output_file):
        """

        :param orig_file: original file including german article title
        :param output_file: output file including translated article title (in English)
        """
        trans_lines_text = self.text_trans(orig_file)
        self.comb_title_eng(orig_file, trans_lines_text, output_file)


class Data_Preprocessing_API:
    data_folder = r"../data"
    os.path.isdir(data_folder)
    train_01, train_02, evaluation, test = "content2019-01-v3.tsv", "content2019-02-v3.tsv", \
                                           "content2019-03-v3.tsv", "MediaEvalNewsImagesBatch04images.tsv"
    processed_data_folder = r"processed/data"

    def load_img(self):
        """
        crawl images from website to folder
        """
        Data_Preprocessing().load_img(os.path.join(self.processed_data_folder, "train.tsv"), "img/training", 3, 4)
        Data_Preprocessing().load_img(os.path.join(self.processed_data_folder, "train_eval.tsv"), "img/train_eval", 3,
                                      4)
        Data_Preprocessing().load_img(os.path.join(self.data_folder, self.test), "img/test", 0, 1)

    def reformat_data(self):
        """
        reformatting the file, the output file only include the selected features
        """
        Data_Preprocessing().reformat_data_file(os.path.join(self.processed_data_folder, "train.tsv"), \
                                                os.path.join(self.processed_data_folder, "train_title.tsv"))
        Data_Preprocessing().reformat_data_file(os.path.join(self.processed_data_folder, "train_eval.tsv"), \
                                                os.path.join(self.processed_data_folder, "train_eval_title.tsv"))

    def translate_title(self):
        """
        tranlate German news article title into English
        """
        Data_Preprocessing().trans_title(os.path.join(self.processed_data_folder, "train_title.tsv"), \
                                         os.path.join(self.processed_data_folder, "train_title_eng.tsv"), 0, 1)
        Data_Preprocessing().trans_title(os.path.join(self.processed_data_folder, "eval_title.tsv"), \
                                         os.path.join(self.processed_data_folder, "eval_title_eng.tsv"), 0, 1)
        Data_Preprocessing().trans_title(os.path.join(self.data_folder, "MediaEvalNewsImagesBatch04articles.tsv"), \
                                         os.path.join(self.processed_data_folder, "test_title_eng.tsv"), 0, 4)