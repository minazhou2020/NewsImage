import os
from URL_Matching import URL_Matching_API
class Model_Ensembling:
    # convert a file into result from image captioning based model
    def cal_caption_result(self, caption_sim_wmd_file):
        lines_caption_sim = []
        with open(caption_sim_wmd_file) as f:
            lines_caption_sim = f.readlines()
        ar_cap_sim_dic = {}
        for line in lines_caption_sim:
            segs = line.strip().split("\t")
            if segs[0] not in ar_cap_sim_dic:
                ar_cap_sim_dic[segs[0]] = {}
            ar_cap_sim_dic[segs[0]][os.path.splitext(segs[1])[0]] = 1 - float(segs[2])
        return ar_cap_sim_dic

    # convert a file into result from face name matching based model
    def face_matching_result(self, face_matching_file):
        image_train_sim = []
        with open(face_matching_file) as f:
            image_train_sim = f.readlines()
        ar_train_sim_dic = {}
        for line in image_train_sim:
            segs = line.strip().split("\t")
            if segs[0] not in ar_train_sim_dic:
                ar_train_sim_dic[segs[0]] = []
            ar_train_sim_dic[segs[0]].append((os.path.splitext(segs[1])[0], segs[2], segs[3]))
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
        return ar_train_sim_dic_cal

    # normalize value of a given dictionary
    def norm_dict(self, a_dict):
        result = {}
        amin, amax = min(a_dict.values()), max(a_dict.values())
        for k, v in a_dict.items():
            if amax - amin == 0:
                result[k] = 0
            else:
                result[k] = (v - amin) / (amax - amin)
        return result

    # normalize value of a given dictionary
    # which is the result from model
    def norm_sim(self, a_dict):
        result = {}
        print(len(a_dict))
        for k, v in a_dict.items():
            result[k] = self.norm_dict(v)
        return result

    #sort and normalize value of a given dictionary
    def sort_dict(self, a_dict):
        normalized_dict = self.norm_sim(a_dict)
        result = {}
        for k, v in normalized_dict.items():
            sort_v = dict(sorted(v.items(), key=lambda item: item[1], reverse=True))
            result[k] = sort_v
        return result

    # merge results from image captioning based model and results from face matching
    # based model
    def merge_cap_face(self, cap_dict, train_dict, crawl_dict, weight_cap, weight_img):
        result = {}
        for k, v in cap_dict.items():
            img_id = os.path.splitext(k)[0]
            result[img_id] = v
            if k in train_dict:
                for k_tr, v_tr in train_dict[k].items():
                    result[img_id][k_tr] = v_tr * weight_img
            if k in crawl_dict:
                for k_cr, v_cr in crawl_dict[k].items():
                    result[img_id][k_cr] = v_cr * weight_img
        return result

    #truncate result into top 100 matching image candidates
    def truncate_result(self, a_dict):
        for key, value in a_dict.items():
            l = [*value]
            l_100 = l[0:100]
            a_dict[key] = l_100

    # merge result from url matching based model into result
    def merge_url(self, final_cap_face_result, url_result):
        final_result = {}
        for s_id, s_value in final_cap_face_result.items():
            if s_id in url_result:
                diff_elements = [x for x in url_result[s_id] if x not in s_value]
                common_elements = [x for x in s_value if x in url_result[s_id]]
                tail_elements = s_value[len(s_value) - len(diff_elements):]
                common_ele_in_tail = [x for x in common_elements if x in tail_elements]
                if len(common_ele_in_tail) > 0:
                    new_value = diff_elements + common_ele_in_tail + s_value[:len(s_value) - len(diff_elements) - len(
                        common_ele_in_tail)]
                else:
                    new_value = diff_elements + s_value[:len(s_value) - len(diff_elements)]
                final_result[s_id] = new_value

            else:
                final_result[s_id] = s_value
        return final_result

    #save final result into a file
    def save_final_result(self, output_file, final_result):
        with open(output_file, "w") as the_file:
            header = "particleID"
            for i in range(100):
                header += "\t" + "iid" + str(i + 1)
            the_file.write(header + "\n")
            for key, value in final_result.items():
                the_file.write(key + '\t' + "\t".join(value) + "\n")

    #ensembling results from mutiple models
    def ensemble_model(self, cap_result, face_tr_result, face_crawl_result,
                       weight_cap, weight_img, url_result, final_output_result):
        ar_cap_sim_dic = self.cal_caption_result(cap_result)
        ar_train_sim_dic_cal = self.face_matching_result(face_tr_result)
        ar_crawl_sim_dic_cal = self.face_matching_result(face_crawl_result)
        cap_dict = self.sort_dict(ar_cap_sim_dic)
        train_dict = self.sort_dict(ar_train_sim_dic_cal)
        crawl_dict = self.sort_dict(ar_crawl_sim_dic_cal)
        cap_face_result = self.merge_cap_face(cap_dict, train_dict, crawl_dict,
                                              weight_cap, weight_img)
        sorted_cap_face_result = self.sort_dict(cap_face_result)
        self.truncate_result(sorted_cap_face_result)
        final_result = self.merge_url(sorted_cap_face_result, url_result)
        self.save_final_result(final_output_result, final_result)

