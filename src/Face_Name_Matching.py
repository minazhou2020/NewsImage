from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import os
import icrawler
from icrawler.builtin import GoogleImageCrawler
import shutil
from shutil import copyfile
import cv2
import time
from datetime import datetime
import os
from deepface import DeepFace



class Face_Name_Matching:
    java_path = r"C:\Users\yuxia\Documents\java-se-8u41-ri\bin\java.exe"
    os.environ['JAVAHOME'] = java_path
    st = StanfordNERTagger(
        r'C:\Users\yuxia\Downloads\stanford-ner-4.2.0\stanford-ner-2020-11-17\classifiers\english.all.3class.distsim.crf.ser.gz',
        r'C:\Users\yuxia\Downloads\stanford-ner-4.2.0\stanford-ner-2020-11-17\stanford-ner.jar',
        encoding='utf-8')

    def concat_name(self, classified_text):
        """
        concate first name and last name into full name
        :param classified_text: text labeling with classes
        :return: list of extracted name
        """
        i = 0
        name_list = []
        while i < len(classified_text) - 1:
            if classified_text[i][1] == 'PERSON':
                name = classified_text[i][0]
                if classified_text[i + 1][1] == 'PERSON':
                    name += " " + classified_text[i + 1][0]
                    i += 1
                name_list.append(name)
            i += 1
        if i == len(classified_text) - 1 and classified_text[i][1] == 'PERSON':
            name_list.append(classified_text[i][0])
        return name_list

    def add_title_name(self, tr_file, output_file):
        """
        add recognized person's name to the input file, and save into another file
        :param tr_file: file including English news article title
        :param output_file: file including person's name recognized from English news article title
        """
        a_file = open(tr_file, encoding="utf8")
        next(a_file)
        cnt = 0
        header = "img_id" + "\t" + "title" + "\t" + "title_eng" + "\t" + "title_names"
        with open(output_file, 'a', encoding="utf-8") as the_file:
            for line in a_file:
                title_eng = line.split("\t")[2]
                tokenized_text = word_tokenize(title_eng)
                classified_text = self.st.tag(tokenized_text)
                names = self.concat_name(classified_text)
                if len(names) > 0:
                    names_str = ','.join(names)
                    print(names_str)
                    new_line = line.strip("\n") + "\t" + names_str + "\n"
                    cnt += 1
                else:
                    new_line = line.strip("\n") + "\t " + "\n"
                the_file.write(new_line)

    def create_name_folder(self, name_file, train_face_folder, train_img_folder):
        """

        :param name_file: file including person's name recognized from English news article title
        :param train_face_folder: path for directory including face images in training dataset
        :param train_img_folder: sub-directories of train_face_folder, folder name are extracted
        person's name
        """
        a_file = open(name_file, encoding="utf8")
        next(a_file)
        for line in a_file:
            line = line.strip("\n")
            img_name = line.split("\t")[0]
            names = line.split("\t")[4].rstrip()
            if len(names) > 0:
                path = os.path.join(train_face_folder, names.split(",")[0])
                if not os.path.exists(path):
                    os.makedirs(path)
                if os.path.exists(os.path.join(train_img_folder, img_name)):
                    copyfile(os.path.join(train_img_folder, img_name), os.path.join(path, img_name))

    def create_mapped_folder(self, d, mapped_folder):
        """

        :param d: path for directory including face images in training dataset
        :param mapped_folder: path for naming index directory including face images in training dataset
        :return:
        """
        if not os.path.isdir(mapped_folder):
            os.mkdir(mapped_folder)
        sub_directories = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
        idx_name = {}
        name_idx = {}
        idx = 1
        for sub_dir in sub_directories:
            idx_name[sub_dir] = 'face_' + str(idx)
            name_idx['face_' + str(idx)] = sub_dir
            idx += 1
        sub_full_paths = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
        for sub_dir in sub_full_paths:
            self.mapped_file_folder(sub_dir, mapped_folder, idx_name)
        return idx_name, name_idx

    def mapped_file_folder(self, src, dest, idx_name):
        """
        copy the file from source directory to naming index directory
        :param src: source directory
        :param dest: destination directory
        :param idx_name: name index
        """
        src_files = os.listdir(src)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            dest_folder = os.path.join(dest, idx_name[os.path.basename(src)])
            if not os.path.isdir(dest_folder):
                os.mkdir(dest_folder)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest_folder)

    def detect_face_cv(self, file):
        """
        detect human face using OpenCV
        :param file: path of image file
        :return: if image contains human face
        """
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Read the input image
        img = cv2.imread(file)
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        width, height = gray.shape
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            if w != width or height != h:
                return True
        return False

    def deep_detect_backend(self, file):
        """
        detect human face using various backends
        :param file: path of image file
        :return: if image contains human face
        """
        backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
        c = 0
        for backend in backends:
            try:
                detected_face = DeepFace.detectFace(file, detector_backend=backend)
            except:
                c += 1
        if c == len(backends):
            return False
        else:
            return True

    def deep_detect(self, file):
        """
        detect human face using various models
        :param file: path of image file
        :return: if image contains human face
        """
        models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
        c = 0
        for model in models:
            try:
                detected_face = DeepFace.detectFace(file, model_name=model)
            except:
                c += 1
        if c == len(models):
            return False
        else:
            return True

    def remove_no_face_img(self, path):
        """
        remove non-face images from image directory
        :param path: path of image directories
        :return: list of face image names
        """
        face_img = []
        sub_directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        cnt = 0
        for sub_dir in sub_directories:
            print(os.path.basename(sub_dir))
            files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith('.jpg')]
            for file in files:
                if self.deep_detect(file) or self.deep_detect_backend(file) or self.detect_face_cv(file):
                    face_img.append(os.path.basename(file))
                else:
                    os.remove(file)
        return face_img

    def remove_empty_folder(self, path):
        """
        remove empty sub-directories in the given directory
        :param path: path of directory
        """
        sub_directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        for sub_dir in sub_directories:
            files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith('.jpg')]
            if len(files) == 0:
                shutil.rmtree(sub_dir)

    def find_mapping(self, d):
        """

        :param d: origin image directory
        :return: a tuptle:
        (a dictionary which key is basename of original directory and
        value is basename of naming index directory,
        a dictionary which key is basename of naming index directory,
        value is basename of original directory)
        """
        sub_directories = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
        idx_name = {}
        name_idx = {}
        idx = 1
        for sub_dir in sub_directories:
            idx_name[sub_dir] = 'face_' + str(idx)
            name_idx['face_' + str(idx)] = sub_dir
            idx += 1
        return idx_name, name_idx

    def get_ar_name_list(self, article_file, title_eng_idx):
        """
        get a list of tuples, first element is article id, second element is English title
        from given file
        :param article_file: file including article id and English news article title
        :param title_eng_idx: column index of news article English title
        :return: a list of tuples, first element is article id, second element is English title
        """
        articles_names = open(article_file, 'r', encoding="utf-8")
        next(articles_names)
        lines = [line.strip() for line in articles_names]
        result = []
        for i in range(len(lines)):
            orig_line = lines[i]
            segs = orig_line.split("\t")
            if len(segs) > title_eng_idx and len(segs[len(segs) - 1].strip()) > 0 and segs[
                len(segs) - 1].strip() != 'NA':
                result.append((segs[0], segs[title_eng_idx].split(",")[0]))
        return result

    def craw_missing_images(self, ar_name_list, idx_name, train_mapped_face_path, crawl_face_path):
        if not os.path.isdir(crawl_face_path):
            os.mkdir(crawl_face_path)
        for ar_name in ar_name_list:
            if ar_name[1] in idx_name and ar_name[1] in idx_name and os.path.exists(
                    os.path.join(train_mapped_face_path, idx_name[ar_name[1]])):
                print("found")
            else:

                if not os.path.isdir(os.path.join(crawl_face_path, ar_name[1])):
                    os.mkdir(os.path.join(crawl_face_path, ar_name[1]))
                google_crawler = GoogleImageCrawler(feeder_threads=1, parser_threads=2, downloader_threads=4,
                                                    storage={'root_dir': os.path.join(crawl_face_path, ar_name[1])})
                filters = dict(date=((2019, 1, 1), (2021, 7, 30)))
                google_crawler.crawl(keyword=ar_name[1], filters=filters, max_num=5, file_idx_offset=0)

    def remove_no_face_img_crawl(self, path):
        face_img = []
        sub_directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        cnt = 0
        for sub_dir in sub_directories:
            print(os.path.basename(sub_dir))
            files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith('.jpg')]
            for file in files:
                if os.path.isfile(file) and (self.deep_detect(file) or self.deep_detect_backend(file) or \
                                             self.detect_face_cv(file)):
                    face_img.append(os.path.basename(file))
                elif os.path.isdir(sub_dir):
                    shutil.rmtree(sub_dir)
        return face_img

    def select_face_image(self, src_path, dst_path):
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        files = [os.path.join(src_path, f) for f in os.listdir(src_path) if f.endswith('.jpg')]
        for file in files:
            if self.deep_detect(file) or self.deep_detect_backend(file) or self.detect_face_cv(file):
                shutil.copy(file, dst_path)

    def get_face_similarity(self, face_img_candidate_dir, train_mapped_img_dir, ar_name_list, idx_name):
        cnt = 0
        record = 0
        img_files = [f for f in os.listdir(face_img_candidate_dir) if f.endswith('.jpg')]
        print(len(img_files))
        ar_img_files = {}
        for ar_name in ar_name_list:
            if ar_name[1].strip() != 'NA' and ar_name[1] in idx_name:
                img_db_path = ""
                if os.path.exists(os.path.join(train_mapped_img_dir, idx_name[ar_name[1]])):
                    img_db_path = os.path.join(train_mapped_img_dir, idx_name[ar_name[1]])
                if len(img_db_path) > 0:
                    df_results = []
                    t = time.process_time()
                    count = 0
                    for img_file in img_files:
                        img_path = os.path.join(face_img_candidate_dir, img_file)
                        df = DeepFace.find(img_path=img_path, db_path=img_db_path,
                                           model_name='Facenet', enforce_detection=False)
                        if len(df) > 0:
                            df_results.append((img_path, df['Facenet_cosine'].mean(), len(df)))
                        else:
                            df_results.append((img_path, "NA", 0))
                        count += 1
                    ar_img_files[ar_name[0]] = df_results
                    cnt += 1
                    elapsed_time = time.process_time() - t

                    print(str(datetime.now()))
                    print("in ", elapsed_time, "seconds complete", cnt, " name completed", " compared with ", count,
                          "images")
            record += 1
            print("processing ", record, " files")
        return ar_img_files

    def write_face_matching_similarity(self, output_file, ar_img_files):
        f = open(output_file, "a")
        for key, v in ar_img_files.items():
            for item in v:
                result = key + "\t" + os.path.basename(item[0]) + "\t" + str(item[1]) + "\t" + str(item[2]) + "\n"
                f.write(result)
        f.close()

    def sort_dictionary(self, input_dict):
        result = {}
        for k, v in input_dict.items():
            sort_v = dict(sorted(v.items(), key=lambda item: item[1], reverse=True))
            result[k] = sort_v
        return result

    def cal_face_matching_similarity(self, input_file):
        image_train_sim = []
        with open(input_file) as f:
            image_train_sim = f.readlines()
        ar_train_sim_dic = {}
        for line in image_train_sim:
            segs = line.strip().split("\t")
            if segs[0] not in ar_train_sim_dic:
                ar_train_sim_dic[segs[0]] = []
            ar_train_sim_dic[segs[0]].append((segs[1], segs[2], segs[3]))
        ar_train_sim_dic_cal = {}
        for k, v in ar_train_sim_dic.items():
            if k not in ar_train_sim_dic_cal:
                ar_train_sim_dic_cal[k] = {}
            for item in v:
                if int(item[2]) == 0:
                    sim = 0
                else:
                    sim = (1 - float(item[1])) * int(item[2])
                ar_train_sim_dic_cal[k][item[0]] = sim
        return self.sort_dictionary(ar_train_sim_dic_cal)

    def cal_MR(self, eval_face_matching):
        count = 0
        for key, value in eval_face_matching.items():
            first_tuple_elements = []
            for a_tuple in value:
                first_tuple_elements.append(a_tuple)
            if key in first_tuple_elements[0:100]:
                count += 1
        return count


class Face_Name_Matching_API:

    def test(self, face_tr_result, face_crawl_result):
        test_ar_name_list = self.fnm.get_ar_name_list(self.test_title_eng_name_file, 3)
        tr_eval_idx_name, tr_eval_name_idx = self.fnm.find_mapping(self.train_eval_face_folder)
        crawl_tr_eval_idx_name, crawl_tr_eval_name_idx = self.fnm.find_mapping(self.tr_eval_crawl_face)
        test_ar_img_files = self.fnm.get_face_similarity(self.test_candidate,
                                                         self.train_eval_name_idx_folder,
                                                         test_ar_name_list,
                                                         tr_eval_idx_name)
        self.fnm.write_face_matching_similarity(face_tr_result, test_ar_img_files)
        test_ar_img_files_crawl = self.fnm.get_face_similarity(self.test_candidate,
                                                               self.tr_eval_crawl_name_idx,
                                                               test_ar_name_list,
                                                               crawl_tr_eval_idx_name)
        self.fnm.write_face_matching_similarity(face_crawl_result, test_ar_img_files)

    def evaluate(self, eval_face_tr_result, eval_face_crawl_result):
        eval_ar_name_list = self.fnm.get_ar_name_list(self.eval_title_eng_name_file, 4)
        tr_idx_name, tr_name_idx = self.fnm.find_mapping(self.train_face_folder)

        crawl_tr_idx_name, crawl_tr_name_idx = self.fnm.find_mapping(self.tr_crawl_face)
        eval_ar_img_files = self.fnm.get_face_similarity(self.eval_candidate,
                                                         self.train_name_idx_folder, eval_ar_name_list,
                                                         tr_idx_name)
        self.fnm.write_face_matching_similarity(eval_face_tr_result, eval_ar_img_files)
        eval_ar_img_files_crawl = self.fnm.get_face_similarity(self.eval_candidate,
                                                               self.tr_crawl_name_idx, eval_ar_name_list,
                                                               crawl_tr_idx_name)
        self.fnm.write_face_matching_similarity(eval_face_crawl_result, eval_ar_img_files_crawl)
        eval_face_matching = self.fnm.cal_face_matching_similarity(eval_face_tr_result)
        eval_face_matching_crawl = self.fnm.cal_face_matching_similarity(eval_face_crawl_result)
        print(self.fnm.cal_MR(eval_face_matching) / len(eval_face_matching))
        print(self.fnm.cal_MR(eval_face_matching) / 2385)

        print(self.fnm.cal_MR(eval_face_matching_crawl) / len(eval_face_matching))
        print(self.fnm.cal_MR(eval_face_matching_crawl) / 2385)

    def create_train_data(self):
        self.face_name_match()
        self.extract_name()
        self.create_name_folder()
        self.crawl_img()
        self.process_crawl_img()
        self.select_candidate()

    def face_name_match(self):
        self.extract_name()
        self.create_name_folder()
        self.crawl_img()
        self.process_crawl_img()
        self.select_candidate()
        self.cal_face_matching_sim()

    def extract_name(self):
        self.fnm.add_title_name(self.tr_title_eng_file, self.tr_title_eng_name_file)
        self.fnm.add_title_name(self.eval_title_eng_file, self.eval_title_eng_name_file)
        self.fnm.add_title_name(self.test_title_eng_file, self.test_title_eng_name_file)
        filenames = [self.tr_title_eng_name_file, self.eval_title_eng_name_file]
        output_file = self.tr_eval_title_eng_name_file
        self.fnm.combine_files(filenames, output_file, False)

    def create_name_folder(self):
        self.fnm.create_name_folder(self.tr_eval_title_eng_name_file,
                                    self.train_eval_face_folder,
                                    r"img\train")
        self.fnm.create_name_folder(self.tr_eval_title_eng_name_file,
                                    self.train_eval_face_folder,
                                    r"img\eval")
        self.fnm.create_name_folder(self.tr_title_eng_name_file,
                                    self.train_face_folder, r"img\train")
        train_eval_face_img_list = self.fnm.remove_no_face_img(self.train_eval_name_idx_folder)
        self.fnm.remove_empty_folder(self.train_eval_name_idx_folder)
        train_face_img_list = self.fnm.remove_no_face_img(self.train_name_idx_folder)
        self.fnm.remove_empty_folder(self.train_name_idx_folder)

    def crawl_img(self):
        train_ar_name_list = self.fnm.get_ar_name_list(self.tr_title_eng_name_file, 4)
        eval_ar_name_list = self.fnm.get_ar_name_list(self.eval_title_eng_name_file, 4)
        test_ar_name_list = self.fnm.get_ar_name_list(self.test_title_eng_name_file, 3)

        self.fnm.craw_missing_images(train_ar_name_list, {}, "", self.crawl_face)
        self.fnm.craw_missing_images(eval_ar_name_list, self.tr_idx_name, self.train_name_idx_folder,
                                     self.tr_crawl_face)
        self.fnm.craw_missing_images(test_ar_name_list, self.tr_eval_idx_name, self.train_eval_name_idx_folder,
                                     self.tr_eval_crawl_face)

    def process_crawl_img(self):
        self.fnm.create_mapped_folder(self.tr_eval_crawl_face, self.tr_eval_crawl_name_idx)
        self.fnm.create_mapped_folder(self.tr_crawl_face, self.tr_crawl_name_idx)
        self.fnm.create_mapped_folder(self.crawl_face, self.crawl_name_idx)
        self.fnm.remove_no_face_img_crawl(self.tr_eval_crawl_name_idx)
        self.fnm.remove_no_face_img_crawl(self.tr_crawl_name_idx)
        self.fnm.remove_no_face_img_crawl(self.crawl_name_idx)

    def select_candidate(self):
        self.fnm.select_face_image(r'img\train', self.train_candidate)
        self.fnm.select_face_image(r'img\eval', self.eval_candidate)
        self.fnm.select_face_image(r'img\test', self.test_candidate)

    def __init__(self):
        self.fnm = Face_Name_Matching()

        self.tr_title_eng_file = r'processed_data\data\train_title_eng.tsv'
        self.eval_title_eng_file = r'processed_data\data\eval_title_eng.tsv'
        self.test_title_eng_file = r'processed_data\data\test_title_eng.tsv'

        self.tr_title_eng_name_file = r'processed_data\data\train_title_eng_name.tsv'
        self.eval_title_eng_name_file = r'processed_data\data\eval_title_eng_name.tsv'
        self.test_title_eng_name_file = r'processed_data\data\test_title_eng_name.tsv'
        self.tr_eval_title_eng_name_file = r"processed_data\data\train_eval_title_eng_name.tsv"

        self.train_eval_face_folder = r"processed_data\img\train_eval_faces"
        self.train_face_folder = r"processed_data\img\train_eval_faces"
        self.train_eval_name_idx_folder = r"processed_data\img\train_eval_mapped_face"
        self.train_name_idx_folder = r"processed_data\img\train_mapped_face"

        self.crawl_face = r"processed_data\img\crawl_face"
        self.tr_crawl_face = r"processed_data\img\crawl_train_face"
        self.tr_eval_crawl_face = r"processed_data\img\crawl_train_eval_face"
        self.crawl_name_idx = r"processed_data\img\crawl_face_mapped"
        self.tr_crawl_name_idx = r"processed_data\img\crawl_train_face_mapped"
        self.tr_eval_crawl_name_idx = r"processed_data\img\crawl_train_eval_face_mapped"

        self.train_candidate = r'processed_data\img\train_face_candidate'
        self.eval_candidate = r'processed_data\img\eval_face_candidate'
        self.test_candidate = r'processed_data\img\test_face_candidate'
