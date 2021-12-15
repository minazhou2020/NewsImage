from scipy import spatial
import gensim.downloader as api
import os
from datetime import datetime


class Image_Caption:

    def __init__(self):
        # choose from multiple models https://github.com/RaRe-Technologies/gensim-data
        self.model = api.load("word2vec-google-news-300")

    def get_caption(self, caption_file):
        """

        :param caption_file: caption file acquired from pre-trained model
        :return: dictionary whose keys are image id, and values are image caption
        """
        print(caption_file)
        articles_names = open(caption_file, 'r', encoding="utf-8")
        lines = [line.strip() for line in articles_names]
        result_dict = {}
        for i in range(len(lines)):
            orig_line = lines[i]
            segs = orig_line.split("\t")
            if segs[0] not in result_dict:
                result_dict[segs[0]] = segs[1]
        return result_dict

    def get_ar_id_title(self, article_file, title_eng_idx):
        """

        :param article_file: article file
        :param title_eng_idx: column index of English title in the file
        :return:dictionary whose keys are article id, values are tuples
         first element is image id, second element is wmd similarity
        """
        print(article_file)
        articles_names = open(article_file, 'r', encoding="utf-8")
        next(articles_names)
        lines = [line.strip() for line in articles_names]
        result_dict = {}
        for i in range(len(lines)):
            orig_line = lines[i]
            segs = orig_line.split("\t")
            if len(segs) >= 3 and segs[0] not in result_dict:
                result_dict[segs[0]] = segs[title_eng_idx]
        return result_dict

    def cal_sim(self, id_title, caption_dict):
        """

        :param id_title: dictionary whose keys are article id, and values are article title
        :param caption_dict: dictionary whose keys are image id, and values are image caption
        :return: dictionary whose keys are article id, values are tuples
         first element is image id, second element is wmd similarity
        """
        sim_result = {}
        cnt = 0
        for ar_id, title in id_title.items():
            caption_sim = []
            for img_id, caption in caption_dict.items():
                sim = self.model.wmdistance(title, caption)
                caption_sim.append((img_id, sim))
            cnt += 1
            sim_result[ar_id] = caption_sim
            # print(str(datetime.now()))
            if cnt % 100 == 0:
                print("complete ", cnt, " article")
        return sim_result

    def write_wmd_sim(self, wmd_sim_file, sim_result):
        """

        :param wmd_sim_file: output file path
        :param sim_result: dictionary whose keys are article id, values are tuples
         first element is image id, second element is wmd similarity
        """
        f = open(wmd_sim_file, "a")
        for key, v in sim_result.items():
            for item in v:
                result = key + "\t" + os.path.basename(item[0]) + "\t" + str(item[1]) + "\n"
                f.write(result)
        f.close()

    def img_cap_similarity(self, caption_file, title_file, result_file, title_eng_idx):
        """

        :param caption_file: caption file acquired from pre-trained model
        :param title_file: article file
        :param result_file: output tsv file
        :param title_eng_idx: column index of English title in the file
        :return:
        """
        caption_dict = self.get_caption(caption_file)
        ar_id_title = self.get_ar_id_title(title_file, title_eng_idx)
        sim_result = self.cal_sim(ar_id_title, caption_dict)
        sorted_sim_result = self.sort_dict(sim_result)
        self.write_wmd_sim(result_file, sorted_sim_result)
        return sorted_sim_result

    def sort_dict(self, sim_result):
        """

        :param sim_result: a dictionary (intermediate result from image caption based method)
        :return: a dictionary which value is a list of tuple sorting by the value of second element
        """
        result = {}
        for k, v in input.items():
            sort_v = dict(sorted(v.items(), key=lambda item: item[1]))
            result[k] = sort_v
        return result

    def eval_cap_sim(self, sort_final_result):
        """

        :param sort_final_result: final result from image caption based method
        :return: MR100
        """
        count = 0
        for key, value in sort_final_result.items():
            first_tuple_elements = []
            for a_tuple in value:
                first_tuple_elements.append(a_tuple)
            if key in first_tuple_elements[0:100]:
                count += 1
        return count / len(sort_final_result)


class Image_Cap_API:
    def eval(self, eval_cap_output):
        """
        Evaluate the performance of image captioning based method on evaluation dataset
        :param eval_cap_output: file path of evaluation result from image captioning based method
        """
        img_cap = Image_Caption()
        eval_result = img_cap.img_cap_similarity(r"processed_data\data\eval_image_caption_result.txt", \
                                                 r"processed_data\data\eval_title_eng.tsv", \
                                                 eval_cap_output, 3)
        img_cap.eval_cap_sim(eval_result)

    def test(self, test_cap_output):
        """
         Make prediction using image captioning based method on evaluation dataset on test dataset
        :param test_cap_output: file path of predictions from image captioning based method
        """
        img_cap = Image_Caption()
        test_result = img_cap.img_cap_similarity(r"processed_data\data\test_image_caption_result.txt", \
                                                 r"processed_data\data\test_title_eng.tsv", \
                                                 test_cap_output, 2)
