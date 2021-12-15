from Data_Preprocessing import Data_Preprocessing
import os


class URL_Matching:
    TR_A_ID_IDX = 1
    TR_I_ID_IDX = 4
    TR_IMG_URL_IDX = 3
    TR_TITLE_IDX = 2

    TEST_A_ID_IDX = 0
    TEST_I_ID_IDX = 1
    TEST_IMG_URL_IDX = 0
    TEST_TITLE_IDX = 2

    def __init__(self):
        self.stop_words = ['in', 'der', 'die', 'und', 'im', 'auf', 'mit', 'fuer', 'von', 'den', 'an', 'fc', 'das', 'am',
                           'vor', 'aus', 'dem', 'anfang', 'sich', 'bei', 'ein', 'des', 'zu', 'sind', 'eine', 'ueber',
                           'gegen', 'nach', 'ist', 'zum', 'beim', 'wird', 'nrw', 'nicht', 'als', 'mehr', 'ab', 'zur',
                           'werden', 'hat', 's', 'wie', 'einem', 'auch', 'e', 'unter', 'wieder', 'vom', 'so', 'um',
                           'noch', 'will', 'afd', 'war', 'strasse']
        self.train_file = 'combine_train.tsv'
        self.data_folder = r'C:\Users\yuxia\Documents\CS7311_Project\FIREWHEEL\data'
        self.eval = "content2019-03-v3.tsv"
        self.test_img_file = "MediaEvalNewsImagesBatch04images.tsv"
        self.test_article_file = "MediaEvalNewsImagesBatch04articles.tsv"

    def extract_gt(self, gt_file):
        """

        :param gt_file: file including ground truth, article id and corresponding image id
        :return: dictionary which key is article id, value is matched image id
        """
        ground_truth = {}
        with open(gt_file, encoding='utf-8') as file:
            next(file)
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            for line in lines:
                segs = line.split("\t")
                if len(segs) < 3:
                    break
                ar_id = segs[self.TR_A_ID_IDX]
                img_id = segs[self.TR_I_ID_IDX]
                ground_truth[ar_id] = img_id
        return ground_truth

    def extract_img_url_token(self, img_url_file, id_idx, img_url_idx):
        """

        :param img_url_file: file including news image url
        :param id_idx: column index of image id in img_url_file
        :param img_url_idx: column index of image url in img_url_file
        :return: dictionary which key is image id, value is token from image url
        """
        img_id_name_dict = {}
        with open(img_url_file, encoding='utf-8') as file:
            next(file)
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            for line in lines:
                segs = line.split("\t")
                if len(segs) < 3:
                    break
                img_id = segs[id_idx]
                img_name_full = segs[img_url_idx].split("/")
                img_name = img_name_full[len(img_name_full) - 1]
                tokens = img_name.split(".")[0].split("-")
                tokens = [item for item in tokens if item.isalpha() and item != "null"]
                img_id_name_dict[img_id] = tokens
        return img_id_name_dict

    def extract_article_token(self, article_file, a_id_idx, ar_name_idx):
        """

        :param article_file: file including news article info
        :param a_id_idx: column index of article id in article_file
        :param ar_name_idx: column index of article url in article_file
        :return: dictionary which key is article id, value is token from article url
        """
        art_id_name_dict = {}
        with open(article_file, encoding="utf8") as file:
            next(file)
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            for line in lines:
                segs = line.split("\t")
                if len(segs) < 3:
                    break
                ar_id = segs[a_id_idx]
                ar_name_full = segs[ar_name_idx].split("/")
                ar_name = ar_name_full[len(ar_name_full) - 1]
                tokens = ar_name.split(".")[0].split("-")
                tokens = [item for item in tokens if item.isalpha() and item != "null"]
                art_id_name_dict[ar_id] = tokens
            return art_id_name_dict

    def match_url(self, art_id_name_dict, img_id_name_dict):
        """
        find the matched images through url comprison
        :param art_id_name_dict: dictionary which key is article id, value is token in article url
        :param img_id_name_dict: dictionary which key is image id, value is token in image url
        :return:
        """
        candidates = {}
        total = 0
        result = {}
        for art_k, art_v in art_id_name_dict.items():
            cnt = 0
            flag = False

            for img_k, img_v in img_id_name_dict.items():
                common_elements = [x for x in art_v if x in img_v and x not in self.stop_words and len(x) > 1]
                if len(common_elements) > 0:
                    if art_k not in result:
                        result[art_k] = []
                    result[art_k].append((img_k, len(common_elements)))
                    flag = True
                    cnt += 1
                    for ele in common_elements:
                        if ele not in candidates:
                            candidates[ele] = 0
                        candidates[ele] += 1
            if art_k in result:
                temp_list = result[art_k]
                temp_list.sort(key=lambda x: x[1], reverse=True)
                result[art_k] = [i[0] for i in temp_list]
            if flag:
                total += 1
        print(total)
        return result

    def write_url_sim(self, result_file, result):
        """
        writing result from url matching based method into a file
        :param result_file: path of file storing result from url matching based method
        :param result:  result from url matching based method
        """
        with open(result_file, 'a') as the_file:
            for art_id, image_list in result.items():
                line = art_id
                for image in image_list:
                    line += "\t" + image
        the_file.close()

    def evaluation(self, result, ground_truth):
        """

        :param result: evaluation result
        :param ground_truth: ground truth
        :return:
        """
        count = 0
        total = 0
        for ar_id, img_id in ground_truth.items():
            if ar_id in result:
                if img_id in result[ar_id][0:100]:
                    count += 1
            total += 1
        return count / total


class URL_Matching_API:
    url_matching = URL_Matching()
    test_img_file = "MediaEvalNewsImagesBatch04images.tsv"
    test_article_file = "MediaEvalNewsImagesBatch04articles.tsv"
    data_folder = r'C:\Users\yuxia\Documents\CS7311_Project\FIREWHEEL\data'

    def test(self, url_pred_file):
        """
        Make prediction using url matching based method on evaluation dataset on test dataset
        url_pred_file: file path of predictions from url matching based method
        """
        img_id_name_dict = URL_Matching.extract_img_url_token(os.path.join(self.data_folder,
                                                                           self.test_img_file),
                                                              URL_Matching.TEST_I_ID_IDX,
                                                              URL_Matching.TEST_IMG_URL_IDX)
        article_id_name_dict = URL_Matching.extract_article_token(os.path.join(self.data_folder,
                                                                               self.test_article_file),
                                                                  URL_Matching.TEST_A_ID_IDX,
                                                                  URL_Matching.TEST_TITLE_IDX)
        result = URL_Matching.match_url(article_id_name_dict, img_id_name_dict)
        URL_Matching.write_url_sim(url_pred_file, result)
        return result

    def evaluation(self, url_eval_file):
        """
        Evaluate the performance of url matching based method on evaluation dataset
        :param url_eval_file: file path of evaluation result from url matching based method
        """
        evaluation_file = 'content2019-03-v3.tsv'
        tr_file = os.path.join(self.data_folder, evaluation_file)
        ground_truth = URL_Matching.extract_gt(tr_file)
        img_id_name_dict = URL_Matching.extract_img_url_token(tr_file,
                                                              URL_Matching.TR_I_ID_IDX,
                                                              URL_Matching.TR_IMG_URL_IDX)
        article_id_name_dict = URL_Matching.extract_article_token(tr_file,
                                                                  URL_Matching.TR_A_ID_IDX,
                                                                  URL_Matching.TR_TITLE_IDX)
        print(len(article_id_name_dict))
        result = URL_Matching.match_url(article_id_name_dict, img_id_name_dict)
        URL_Matching.write_url_sim(url_eval_file, result)
        evaluation_result = URL_Matching.evaluation(result, ground_truth)
        print("MR100 in evaluation dataset is ", evaluation_result)
