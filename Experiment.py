from Data_Preprocessing import Data_Preprocessing_API
from Face_Name_Matching import Face_Name_Matching_API
from URL_Matching import URL_Matching_API
import Model_Ensembling
import os
from Image_Caption import Image_Cap_API


def create_folder(path):
    """
    create a folder if not exist
    :param path: path of the folder
    """
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)


def create_all_folder():
    """
    create required folders
    """
    create_folder('processed_data')
    create_folder('img')
    create_folder(r'processed_data/data')
    create_folder(r'processed_data/img')
    create_folder('result')


def data_prepare():
    """
    data preparation
    """
    dpi = Data_Preprocessing_API()
    dpi.load_img()
    dpi.reformat_data()
    dpi.translate_title()
    Face_Name_Matching_API().create_train_data()


def eval(eval_face_tr_result, eval_face_crawl_result, eval_cap_result, eval_url_output):
    """
    evaluating the model performance on evaluation data
    :param eval_face_tr_result: file path of evaluation result from face name matching result
    (model is trained with face images in training dataset)
    :param eval_face_crawl_result: file path of evaluation result from face name matching result
    (model is trained with crawled face images)
    :param eval_cap_result: file path of evaluation result from image caption based method
    :param eval_url_output: file path of evaluation result from url matching based method
    """
    Face_Name_Matching_API().evaluate(eval_face_tr_result, eval_face_crawl_result)
    URL_Matching_API().evaluate(eval_url_output)
    Image_Cap_API().evaluate(eval_cap_result)


def predict(face_tr_result, face_crawl_result, test_cap_result, test_url_output,
            weight_cap, weight_img, final_output_result):
    """

    :param face_tr_result: file path of prediction from face name matching result
    (model is trained with face images in training dataset)
    :param face_crawl_result: file path of prediction from face name matching result
    (model is trained with crawled face images)
    :param test_cap_result: file path of prediction from image caption based method
    :param test_url_output: file path of prediction from url matching based method
    :param weight_cap: ensembling weights of image caption based model
    :param weight_img: ensembling weights of face-name matching based model
    :param final_output_result: file path of final result
    """
    Face_Name_Matching_API().test(face_tr_result, face_crawl_result)
    test_url_result = URL_Matching_API().test(test_url_output)
    Image_Cap_API().test(test_cap_result)
    Model_Ensembling().ensemble_model(test_cap_result, face_tr_result, face_crawl_result,
                                      weight_cap, weight_img, test_url_result, final_output_result)
