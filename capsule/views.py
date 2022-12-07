from random import random

from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
import tensorflow.keras.backend as K
from tensorflow import keras
from matplotlib import pyplot
import matplotlib
matplotlib.use('Agg')
import os
import json
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import cv2
import torch
import torchvision

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from skimage.metrics import structural_similarity

import glob
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import iqr
from PIL import Image


#***************************** Capsule Model *******************************#

def resmaps_ssim(imgs_input, imgs_pred):
    resmaps = np.zeros(shape=imgs_input.shape, dtype="float64")
    scores = []
    for index in range(len(imgs_input)):
        img_input = imgs_input[index]
        img_pred = imgs_pred[index]
        score, resmap = structural_similarity(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            multichannel=False,
            sigma=1.5,
            full=True,
        )
        # resmap = np.expand_dims(resmap, axis=-1)
        resmaps[index] = 1 - resmap
        scores.append(score)
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    return scores, resmaps


def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred) ** 2
    scores = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    return scores, resmaps


def calculate_resmaps(imgs_input, imgs_pred, method, dtype="float64"):
    """
    To calculate resmaps, input tensors must be grayscale and of shape (samples x length x width).
    """
    # if RGB, transform to grayscale and reduce tensor dimension to 3
    if imgs_input.ndim == 4 and imgs_input.shape[-1] == 3:
        imgs_input_gray = tf.image.rgb_to_grayscale(imgs_input).numpy()[:, :, :, 0]
        imgs_pred_gray = tf.image.rgb_to_grayscale(imgs_pred).numpy()[:, :, :, 0]
    else:
        imgs_input_gray = imgs_input
        imgs_pred_gray = imgs_pred

    # calculate remaps
    if method == "l2":
        scores, resmaps = resmaps_l2(imgs_input_gray, imgs_pred_gray)
    elif method in ["ssim", "mssim"]:
        scores, resmaps = resmaps_ssim(imgs_input_gray, imgs_pred_gray)
    if dtype == "uint8":
        resmaps = img_as_ubyte(resmaps)
    return scores, resmaps


# losses.py
def ssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):

        # return (1 - tf.image.ssim(imgs_true, imgs_pred, dynamic_range)) / 2

        return 1 - tf.image.ssim(imgs_true, imgs_pred, dynamic_range)

        # return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, dynamic_range))

    return loss


def mssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):

        return 1 - tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)

        # return 1 - tf.reduce_mean(
        #     tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)
        # )

    return loss


def l2_loss(imgs_true, imgs_pred):
    # return 2 * tf.nn.l2_loss(imgs_true - imgs_pred)
    return tf.nn.l2_loss(imgs_true - imgs_pred)


# metrics.py
def ssim_metric(dynamic_range):
    def ssim(imgs_true, imgs_pred):
        return K.mean(tf.image.ssim(imgs_true, imgs_pred, dynamic_range), axis=-1)

    return ssim


def mssim_metric(dynamic_range):
    def mssim(imgs_true, imgs_pred):
        return K.mean(
            tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range), axis=-1
        )
    return mssim


# custom function to load model
def get_model_info(model_path):
    dir_name = os.path.dirname(model_path)
    print(os.getcwd())
    with open(f"{os.getcwd()}\\capsule/model/info.json", "r") as read_file:
        info = json.load(read_file)
    return info


def load_model_HDF5(model_path):
    """
    Loads model (HDF5 format), training setup and training history.
    """

    # load parameters
    info = get_model_info(model_path)
    loss = info["model"]["loss"]
    dynamic_range = info["preprocessing"]["dynamic_range"]

    # load autoencoder
    if loss == "mssim":
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "loss": mssim_loss(dynamic_range),
                "mssim": mssim_metric(dynamic_range),
            },
            compile=True,
        )

    elif loss == "ssim":
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "loss": ssim_loss(dynamic_range),
                "ssim": ssim_metric(dynamic_range),
            },
            compile=True,
        )

    else:
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "l2_loss": l2_loss,
                "ssim": ssim_loss(dynamic_range),
                "mssim": mssim_metric(dynamic_range),
            },
            compile=True,
        )

    # load training history
    dir_name = os.path.dirname(model_path)
    history = pd.read_csv(os.path.join(dir_name, "history.csv"))

    return model, info, history


"""### load images as tensors in dict"""

import copy
from pathlib import Path


def get_img_dict(directory: Path):
    tensor_dict = {}

    for dir_ in directory.iterdir():
        dir_name = dir_.parts[-1]
        # print(dir_, dir_name)
        curr_imgs = glob.glob(f"{dir_}/*.png")
        curr_np_arr = np.array([
            np.array(Image.open(fname).resize((256, 256)))
            for fname in curr_imgs
        ])
        curr_np_arr = curr_np_arr.astype(np.float32)
        curr_np_arr /= 255.
        print(dir_name, curr_np_arr.shape)
        tensor_dict[dir_name] = curr_np_arr

    return tensor_dict


def get_img_dict_mutiple(directory: Path):
    tensor_dict = {}

    for dir_ in directory.iterdir():
        dir_name = dir_.parts[-1]


        print(dir_, dir_name)
        curr_imgs = dir_

        curr_np_arr = np.array([
            np.array(Image.open(dir_).resize((256, 256)))
        ])
        curr_np_arr = curr_np_arr.astype(np.float32)
        curr_np_arr /= 255.
        img_name = curr_imgs
        print(dir_name, curr_np_arr.shape)
        tensor_dict[dir_name] = curr_np_arr

    return tensor_dict


def get_img_dict_input(directory: Path):
    tensor_dict = {}
    dir_name=directory.name
    fname=(dir_name.split("."))[0]
    curr_np_arr = np.array([
        np.array(Image.open(directory).resize((256, 256)))
    ])
    curr_np_arr = curr_np_arr.astype(np.float32)
    curr_np_arr /= 255.
    tensor_dict[fname] = curr_np_arr
    print(dir_name, curr_np_arr.shape)
    return tensor_dict


def create_in_pred_res_dict(model, images_dict):
    dict_tuples = {}

    for k, imgs in images_dict.items():
        curr_pred = model.predict(imgs)
        curr_resmap = calculate_resmaps(imgs, curr_pred, method="l2")[1]
        dict_tuples[k] = (imgs, curr_pred, curr_resmap)

    return dict_tuples


# returned from create_in_pred_res_dict
def create_images_batch_from_dict(input_gen_res_data_dict):
    all_merged_inp = []
    all_merged_gen = []
    all_merged_res = []
    all_merged_labels = []

    for lbl, np_arr in input_gen_res_data_dict.items():
        input_img, gen_img, res_map = np_arr
        all_merged_inp.append(input_img)
        all_merged_gen.append(gen_img)
        all_merged_res.append(res_map)

        all_merged_labels.extend([lbl] * input_img.shape[0])

    all_merged_inp = np.vstack(all_merged_inp)
    all_merged_gen = np.vstack(all_merged_gen)
    all_merged_res = np.vstack(all_merged_res)

    all_merged_data = (all_merged_inp, all_merged_gen, all_merged_res)
    print(all_merged_inp.shape, all_merged_gen.shape, all_merged_res.shape, len(all_merged_labels))

    return all_merged_data, all_merged_labels


from scipy import stats as st


def capsules_prediction(cap_model,
                        data_samples: np.array,
                        data_samples_input: np.array,
                        data_labels_input: list,
                        data_labels: list,
                        capsules_data_dict: dict,
                        capsules_data_dict_input: dict,
                        Z_FACTOR: float = 1.96, # 95 %CI, use 2.57 for 99% CI
                        ):

    #limit = min(limit, data_samples.shape[0])
    HISTOGRAM_BINS = 25
    IMG_CROP_DIM = (75, 170)
    VALS_FILTER = 0.01
    CUT_POINT = 130 # for left and right partition

    curr_imgs = data_samples_input
    if len(data_labels_input) == 0:
        data_labels_input = ["TEST"] * curr_imgs.shape[0]
    labels = data_labels_input
    assert curr_imgs.shape[0] == len(labels), "data points and labels should match"

    generated_images = cap_model.predict(curr_imgs)
    _, residual_maps = calculate_resmaps(curr_imgs, generated_images, method="l2")

    #curr_imgs, generated_images, residual_maps = curr_imgs[:limit], generated_images[:limit], residual_maps[:limit]
    # print(type(curr_imgs), type(generated_images), type(residual_maps))

    idx = 0
    temp_tbl = {"img":[], # input/good

                "left_vals_count":[],
                "left_mean":[],
                "left_std":[],
                "left_IQR":[],
                "left_min":[],
                "left_25":[],
                "left_50":[],
                "left_75":[],
                "left_max":[],

                "right_vals_count":[],
                "right_mean":[],
                "right_std":[],
                "right_IQR":[],
                "right_min":[],
                "right_25":[],
                "right_50":[],
                "right_75":[],
                "right_max":[],
                }

    final_stat_dict = {
        "lower_bound_iqr_left":[],
        "upper_bound_iqr_left": [],
        "input_img_iqr_left":[],
        "lift_iqr_left":[],
        "lower_bound_iqr_right":[],
        "upper_bound_iqr_right": [],
        "lift_iqr_right":[],
        "input_img_iqr_right":[],
        "actual_lbl": [],
        "lower_bound_range_right":[],
        "upper_bound_range_right":[],
        "input_img_range_right":[],
        "lower_bound_range_left":[],
        "upper_bound_range_left":[],
        "input_img_range_left":[],
        "prediction": [],
        "prediction_result": [],
        "lift_range_right": [],
        "lift_range_left": []
    }

    plots = []

    for input_img, gen_img, resmap, lbl in zip(curr_imgs, generated_images, residual_maps, labels):
        idx += 1
        # if idx == 5: break
        curr_temp_tbl = copy.deepcopy(temp_tbl)

        # print(type(input_img), type(gen_img), type(resmap))
        # print(input_img.shape, gen_img.shape, resmap.shape)
        # print(IMG_CROP_DIM[0], IMG_CROP_DIM[1])

        input_img = input_img[IMG_CROP_DIM[0]:IMG_CROP_DIM[1], :, :]
        gen_img = gen_img[IMG_CROP_DIM[0]:IMG_CROP_DIM[1], :, :]
        resmap = resmap[IMG_CROP_DIM[0]:IMG_CROP_DIM[1], :]

        fig1, ax1 = plt.subplots(1, 2, figsize=(7, 5))
        ax1[0].imshow(input_img)
        ax1[0].title.set_text("input("+lbl+")")

        # ax1[1].imshow(gen_img)
        # ax1[1].title.set_text("generated")

        ax1[1].imshow(resmap, cmap="plasma") # plasma, inferno, cviridis
        ax1[1].title.set_text("resmap")

        fig1.savefig("media/capsule_fig1/my_resmap_"+lbl)
        # fig1.show()

        fig2, ax2 = plt.subplots(1, 4, figsize=(20, 3))

        in_left_resmap, in_right_resmap = resmap[:, :CUT_POINT], resmap[:, CUT_POINT:]
        in_left, in_right = in_left_resmap.flatten(), in_right_resmap.flatten()
        in_vals_left, in_vals_right = in_left[in_left > VALS_FILTER], in_right[in_right > VALS_FILTER]

        in_vector_left = np.histogram(in_vals_left, bins=HISTOGRAM_BINS, range=(0, 1))
        curr_temp_tbl["img"].append("input")
        curr_temp_tbl["left_vals_count"].append(len(in_vals_left))
        curr_temp_tbl["left_mean"].append(in_vals_left.mean())
        curr_temp_tbl["left_std"].append(in_vals_left.std(ddof=1))
        curr_temp_tbl["left_IQR"].append(iqr(in_vals_left))
        curr_temp_tbl["left_min"].append(in_vals_left.min())
        curr_temp_tbl["left_25"].append(np.quantile(in_vals_left, 0.25))
        curr_temp_tbl["left_50"].append(np.quantile(in_vals_left, 0.5))
        curr_temp_tbl["left_75"].append(np.quantile(in_vals_left, 0.75))
        curr_temp_tbl["left_max"].append(in_vals_left.max())

        in_vector_right = np.histogram(in_vals_right, bins=HISTOGRAM_BINS, range=(0, 1))
        curr_temp_tbl["right_vals_count"].append(len(in_vals_right))
        curr_temp_tbl["right_mean"].append(in_vals_right.mean())
        curr_temp_tbl["right_std"].append(in_vals_right.std(ddof=1))
        curr_temp_tbl["right_IQR"].append(iqr(in_vals_right))
        curr_temp_tbl["right_min"].append(in_vals_right.min())
        curr_temp_tbl["right_25"].append(np.quantile(in_vals_right, 0.25))
        curr_temp_tbl["right_50"].append(np.quantile(in_vals_right, 0.5))
        curr_temp_tbl["right_75"].append(np.quantile(in_vals_right, 0.75))
        curr_temp_tbl["right_max"].append(in_vals_right.max())

        #ax2[0][0].imshow(in_left_resmap, cmap="plasma")
        fitted_data_left, fitted_lambda_left = st.boxcox(in_vals_left)
        ax2[0].title.set_text("in_res_l")
        ax2[0].hist(abs(fitted_data_left), bins=HISTOGRAM_BINS)

        #ax2[0][1].imshow(in_right_resmap, cmap="plasma")
        fitted_data_right, fitted_lambda_right = st.boxcox(in_vals_right)
        ax2[1].title.set_text("in_res_r")
        ax2[1].hist(abs(fitted_data_right), bins=HISTOGRAM_BINS)

        five_random_indexes = np.random.choice(np.arange(capsules_data_dict["good"][0].shape[0]), size=5)
        good_examples = capsules_data_dict["good"][0][five_random_indexes]
        good_examples_gen = cap_model.predict(good_examples)
        _, good_examples_resmaps = calculate_resmaps(good_examples, good_examples_gen, method="l2")
        good_examples_resmaps = good_examples_resmaps[:, IMG_CROP_DIM[0]:IMG_CROP_DIM[1], :]
        good_resmap_left, good_resmap_right = good_examples_resmaps[:, :, :CUT_POINT], good_examples_resmaps[:, :, CUT_POINT:]
        # good_examples_resmaps = resmaps_train[five_random_indexes]
        vals_left_list=[]
        vals_right_list=[]
        labels_left_list=[]
        labels_right_list=[]
        for idx, l_resmap, r_resmap in zip(range(2, 12, 2), good_resmap_left, good_resmap_right):
            curr_temp_tbl["img"].append(f"good_{five_random_indexes[idx//2 - 1]}")
            vals_left, vals_right = l_resmap.flatten(), r_resmap.flatten()
            vals_left, vals_right = vals_left[vals_left > VALS_FILTER], vals_right[vals_right > VALS_FILTER]

            curr_temp_tbl["left_vals_count"].append(len(vals_left))
            curr_temp_tbl["left_mean"].append(vals_left.mean())
            curr_temp_tbl["left_std"].append(vals_left.std(ddof=1))
            curr_temp_tbl["left_IQR"].append(iqr(vals_left))
            curr_temp_tbl["left_min"].append(vals_left.min())
            curr_temp_tbl["left_25"].append(np.quantile(vals_left, 0.25))
            curr_temp_tbl["left_50"].append(np.quantile(vals_left, 0.5))
            curr_temp_tbl["left_75"].append(np.quantile(vals_left, 0.75))
            curr_temp_tbl["left_max"].append(vals_left.max())

            curr_temp_tbl["right_vals_count"].append(len(vals_right))
            curr_temp_tbl["right_mean"].append(vals_right.mean())
            curr_temp_tbl["right_std"].append(vals_right.std(ddof=1))
            curr_temp_tbl["right_IQR"].append(iqr(vals_right))
            curr_temp_tbl["right_min"].append(vals_right.min())
            curr_temp_tbl["right_25"].append(np.quantile(vals_right, 0.25))
            curr_temp_tbl["right_50"].append(np.quantile(vals_right, 0.5))
            curr_temp_tbl["right_75"].append(np.quantile(vals_right, 0.75))
            curr_temp_tbl["right_max"].append(vals_right.max())
            fitted_data_good_left, fitted_lambda_good_left = st.boxcox(vals_left)
            vals_left_list.append(abs(fitted_data_good_left))
            fitted_data_good_right, fitted_lambda_good_right = st.boxcox(vals_right)
            vals_right_list.append(abs(fitted_data_good_right))
            labels_left_list.append(f"g_res_{idx}_l")
            labels_right_list.append(f"g_res_{idx}_r")

        ax2[2].hist(vals_left_list, bins=HISTOGRAM_BINS,label=labels_left_list,histtype ='step')
        # ax2[2].legend(loc='upper right')

        ax2[3].hist(vals_right_list, bins=HISTOGRAM_BINS,label=labels_right_list,histtype ='step')
        # ax2[3].legend(loc='upper right')

        fig2.savefig("media/capsule_fig2/my_graph_"+lbl)
        # AVERAGE OF LAST 5
        curr_temp_tbl["img"].append("avg")
        curr_temp_tbl["left_vals_count"].append(np.mean(curr_temp_tbl["left_vals_count"][-5:]))
        curr_temp_tbl["left_mean"].append(np.mean(curr_temp_tbl["left_mean"][-5:]))
        curr_temp_tbl["left_std"].append(np.mean(curr_temp_tbl["left_std"][-5:]))
        curr_temp_tbl["left_IQR"].append(np.mean(curr_temp_tbl["left_IQR"][-5:]))
        curr_temp_tbl["left_min"].append(np.mean(curr_temp_tbl["left_min"][-5:]))
        curr_temp_tbl["left_25"].append(np.mean(curr_temp_tbl["left_25"][-5:]))
        curr_temp_tbl["left_50"].append(np.mean(curr_temp_tbl["left_50"][-5:]))
        curr_temp_tbl["left_75"].append(np.mean(curr_temp_tbl["left_75"][-5:]))
        curr_temp_tbl["left_max"].append(np.mean(curr_temp_tbl["left_max"][-5:]))

        curr_temp_tbl["right_vals_count"].append(np.mean(curr_temp_tbl["right_vals_count"][-5:]))
        curr_temp_tbl["right_mean"].append(np.mean(curr_temp_tbl["right_mean"][-5:]))
        curr_temp_tbl["right_std"].append(np.mean(curr_temp_tbl["right_std"][-5:]))
        curr_temp_tbl["right_IQR"].append(np.mean(curr_temp_tbl["right_IQR"][-5:]))
        curr_temp_tbl["right_min"].append(np.mean(curr_temp_tbl["right_min"][-5:]))
        curr_temp_tbl["right_25"].append(np.mean(curr_temp_tbl["right_25"][-5:]))
        curr_temp_tbl["right_50"].append(np.mean(curr_temp_tbl["right_50"][-5:]))
        curr_temp_tbl["right_75"].append(np.mean(curr_temp_tbl["right_75"][-5:]))
        curr_temp_tbl["right_max"].append(np.mean(curr_temp_tbl["right_max"][-5:]))

        comparison_df = pd.DataFrame(curr_temp_tbl)
        comparison_df["left_range"] = comparison_df["left_max"] - comparison_df["left_min"]
        comparison_df["right_range"] = comparison_df["right_max"] - comparison_df["right_min"]
        #display(comparison_df)

        good_stats = comparison_df.query("img == 'avg'")[["left_IQR", "right_IQR"]]
        left_iqr_mean = good_stats.loc[:, "left_IQR"].values[0]
        left_iqr_std = np.std(curr_temp_tbl["left_IQR"][-6:-1])
        left_lower_iqr, left_upper_iqr = left_iqr_mean - Z_FACTOR * left_iqr_std / 4, left_iqr_mean + Z_FACTOR * left_iqr_std / 4
        #print(f"99% CI for left IQR - [{left_lower_iqr}, {left_upper_iqr}]")
        #print(f"left_input_iqr - {comparison_df.loc[0, 'left_IQR']}")

        right_iqr_mean = good_stats.loc[:, "right_IQR"].values[0]
        right_iqr_std = np.std(curr_temp_tbl["right_IQR"][-6:-1])
        right_lower_iqr, right_upper_iqr = right_iqr_mean - Z_FACTOR * right_iqr_std / 4, right_iqr_mean + Z_FACTOR * right_iqr_std / 4
        #print(f"99% CI for right IQR - [{right_lower_iqr}, {right_upper_iqr}]")
        #print(f"right_input_iqr - {comparison_df.loc[0, 'right_IQR']}")

        # Min - max + 95,99% CI
        range_mean_left = comparison_df.loc[1:5, "left_range"].mean()
        range_std_left = comparison_df.loc[1:5, "left_range"].std()
        marginal_err_range_left = Z_FACTOR * range_std_left / 4
        left_lower_range, left_upper_range = 0.027, 0.063
        #print(f"99% CI for left RANGE - [{left_lower_range}, {left_upper_range}]")
        #print(f"left_input_range - {comparison_df.loc[0, 'left_range']}")

        range_mean_right = comparison_df.loc[1:5, "right_range"].mean()
        range_std_right = comparison_df.loc[1:5, "right_range"].std()
        marginal_err_range_right = Z_FACTOR * range_std_right / 4
        right_lower_range, right_upper_range = 0.031, 0.058
        #print(f"99% CI for right RANGE - [{right_lower_range}, {right_upper_range}]")
        #print(f"right_input_range - {comparison_df.loc[0, 'right_range']}")

        final_stat_dict["lower_bound_iqr_left"].append(left_lower_iqr)
        final_stat_dict["upper_bound_iqr_left"].append(left_upper_iqr)
        final_stat_dict["input_img_iqr_left"].append(comparison_df.loc[0, 'left_IQR'])
        bound_list_left=[left_lower_iqr,left_upper_iqr]
        closest_left = min(bound_list_left, key=lambda x: abs(x - comparison_df.loc[0, 'left_IQR']))
        if (left_lower_iqr<=comparison_df.loc[0, 'left_IQR']) & (left_upper_iqr>=comparison_df.loc[0, 'left_IQR']):
            lift_left=1
        else:
            lift_left=abs(comparison_df.loc[0, 'left_IQR']-closest_left)+1
        final_stat_dict["lift_iqr_left"].append(lift_left)

        final_stat_dict["lower_bound_iqr_right"].append(right_lower_iqr)
        final_stat_dict["upper_bound_iqr_right"].append(right_upper_iqr)
        final_stat_dict["input_img_iqr_right"].append(comparison_df.loc[0, 'right_IQR'])
        bound_list_right=[right_lower_iqr,right_upper_iqr]
        closest_right = min(bound_list_right, key=lambda x: abs(x - comparison_df.loc[0, 'right_IQR']))
        if (right_lower_iqr<=comparison_df.loc[0, 'right_IQR']) & (right_upper_iqr>=comparison_df.loc[0, 'right_IQR']):
            lift_right=1
        else:
            lift_right=abs(comparison_df.loc[0, 'right_IQR']-closest_right)+1
        final_stat_dict["lift_iqr_right"].append(lift_right)

        #final_stat_dict["lower_bound_range_right"].append(right_lower_range)
        #final_stat_dict["upper_bound_range_right"].append(right_upper_range)
        #final_stat_dict["input_img_range_right"].append(comparison_df.loc[0, 'right_range'])

        final_stat_dict["actual_lbl"].append(lbl)
        final_stat_dict["lower_bound_range_right"].append(0.031)
        final_stat_dict["upper_bound_range_right"].append(0.058)
        final_stat_dict["input_img_range_right"].append(comparison_df.loc[0, 'right_range'])
        final_stat_dict["lower_bound_range_left"].append(0.027)
        final_stat_dict["upper_bound_range_left"].append(0.063)
        final_stat_dict["input_img_range_left"].append(comparison_df.loc[0, 'left_range'])

        if (right_lower_range <= comparison_df.loc[0, 'right_range']) & (right_upper_range >= comparison_df.loc[0, 'right_range']) & (left_lower_range <= comparison_df.loc[0, 'left_range']) & (left_upper_range >= comparison_df.loc[0, 'left_range']):
            final_stat_dict['prediction'].append('Non-Defective')
            final_stat_dict['prediction_result'].append(1)
            final_stat_dict['lift_range_right'].append(1)
            final_stat_dict['lift_range_left'].append(1)
        else:
            final_stat_dict['prediction'].append('Defective')
            final_stat_dict['prediction_result'].append(0)
            bound_list_right=[right_lower_range, right_upper_range]
            closest_right = min(bound_list_right, key=lambda x: abs(x - comparison_df.loc[0, 'right_range']))
            lift_range_right=abs((comparison_df.loc[0, 'right_range']-closest_right)/closest_right) * 100
            bound_list_left=[left_lower_range, left_upper_range]
            closest_left = min(bound_list_left, key=lambda x: abs(x - comparison_df.loc[0, 'left_range']))
            lift_range_left=abs((comparison_df.loc[0, 'left_range']-closest_left)/closest_left) * 100
            final_stat_dict['lift_range_right'].append(lift_range_right)
            final_stat_dict['lift_range_left'].append(lift_range_left)

    final_stats_df = pd.DataFrame(final_stat_dict)

    final_stats_df["left_iqr_within_CI"] = (
            (final_stats_df["lower_bound_iqr_left"] <= final_stats_df["input_img_iqr_left"])
            & (final_stats_df["input_img_iqr_left"] <= final_stats_df["upper_bound_iqr_left"])
    )
    final_stats_df["right_iqr_within_CI"] = (
            (final_stats_df["lower_bound_iqr_right"] <= final_stats_df["input_img_iqr_right"])
            & (final_stats_df["input_img_iqr_right"] <= final_stats_df["upper_bound_iqr_right"])
    )

    final_stats_df["good_left"] = final_stats_df["left_iqr_within_CI"]
    final_stats_df["good_right"] = final_stats_df["right_iqr_within_CI"]

    return final_stats_df, plots


model_path = f"{os.getcwd()}\\capsule/model/mvtecCAE_b8_e24.hdf5"
cap_model, info, _ = load_model_HDF5(model_path)
cap_model.summary()

from django.conf import settings
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from plotly.offline import plot, offline
from plotly.graph_objs import Bar, pie


def upload_capsule(request):
    if request.method == 'POST':
        context = {}
        uploaded_file = request.FILES['file1']

        assessment_portfolio = request.session.get('assessment_portfolio')
        context["assessment_portfolio"] = assessment_portfolio

        file_list = []
        for i in request.FILES.getlist('file1'):
            file_list.append(i)

        if len(file_list) is 1:
            pass
            fs = FileSystemStorage()
            name = fs.save(uploaded_file.name, uploaded_file)
            context["url"] = fs.url(name)
            testimage = '.'+context["url"]
            context['name'] = testimage
            context['test_image'] = testimage

            print(f"Imag url is {testimage}")

            capsule_imgs = get_img_dict(Path(r"./capsule/data"))
            capsules_data_dict = create_in_pred_res_dict(cap_model, capsule_imgs)
            all_merged_capsules, all_merged_capsules_label = create_images_batch_from_dict(capsules_data_dict)
            for idx, k in enumerate(capsules_data_dict.keys()):
                f, h, r = capsules_data_dict[k]
                print(f"{idx, k}\norig - {f.shape}")
                print(f"gene - {h.shape}")
                print(f"resi - {r.shape}")
                print()
    #
        else:
            fs = FileSystemStorage(location='media/capsule')
            for i in file_list:
                path = os.listdir(settings.MEDIA_ROOT_CAPSULE)
                try:
                    if i.name in path:
                        # path.remove(i.name)
                        os.remove('media/capsule/' + i.name)
                except Exception as ex:
                    print(ex)

                fs.save(i.name, i)
                # context["url_{}".format(i)] = fs.url(fs.save(i.name, i))
                # list_of_files = fs.url(fs.save(i.name, i))
                # path = os.path.join(settings.BASE_DIR, 'media\\test\\')

                list_of_files = os.listdir('media/capsule')
                context['list_of_files'] = list_of_files

            print(list_of_files)
            # capsule_imgs = get_img_dict(Path(r"./capsule/data"))
            capsule_imgs = get_img_dict(Path(r"./capsule/data"))
            print(capsule_imgs.keys())
            capsules_data_dict = create_in_pred_res_dict(cap_model, capsule_imgs)
            all_merged_capsules, all_merged_capsules_label = create_images_batch_from_dict(capsules_data_dict)
            for idx, k in enumerate(capsules_data_dict.keys()):
                f, h, r = capsules_data_dict[k]
                print(f"{idx, k}\norig - {f.shape}")
                print(f"gene - {h.shape}")
                print(f"resi - {r.shape}")
                print()

        if assessment_portfolio == '1':
            pass
        else:
            capsule_imgs_input = get_img_dict_mutiple(Path(rf"{'media/capsule'}"))
            capsules_data_dict_input = create_in_pred_res_dict(cap_model, capsule_imgs_input)
            all_merged_capsules_input, all_merged_capsules_label_input = create_images_batch_from_dict(capsules_data_dict_input)
            for idx, k in enumerate(capsules_data_dict_input.keys()):
                f_input, h_input, r_input = capsules_data_dict_input[k]
                print(f"{idx, k}\norig - {f_input.shape}")
                print(f"gene - {h_input.shape}")
                print(f"resi - {r_input.shape}")
                print()

            cpl_df, cpl_plots = capsules_prediction(cap_model, all_merged_capsules[0], all_merged_capsules_input[0], all_merged_capsules_label_input,
                                                    all_merged_capsules_label,
                                                    capsules_data_dict, capsules_data_dict_input,
                                                    Z_FACTOR = 1.96 # 95 %CI, use 2.57 for 99% CI
                                                    )
            print(cpl_df)
            table = json.loads(cpl_df.to_json(orient='records'))
            context['table'] = table

            list_of_values_right = []
            list_of_values_left = []
            lower_bound_left = []
            upper_bound_left = []
            lower_bound_right = []
            upper_bound_right = []
            lift_range_right = []
            lift_range_left = []
            input_img = []
            for i in table:
                list_of_values_right.append(i.get('prediction_result'))
                request.session['list_of_values_right'] = list_of_values_right

            for i in table:
                list_of_values_left.append(i.get('input_img_range_left'))
                request.session['list_of_values_left'] = list_of_values_left

            for i in table:
                lower_bound_left.append(i.get('lower_bound_range_left'))
                request.session['lower_bound_left'] = lower_bound_left

            for i in table:
                lower_bound_right.append(i.get('lower_bound_range_right'))
                request.session['lower_bound_right'] = lower_bound_right

            for i in table:
                upper_bound_left.append(i.get('upper_bound_range_left'))
                request.session['upper_bound_left'] = upper_bound_left

            for i in table:
                upper_bound_right.append(i.get('upper_bound_range_right'))
                request.session['upper_bound_right'] = upper_bound_right

            for i in table:
                input_img.append(i.get('input_img_range_left'))
                request.session['input_img'] = input_img

            for i in table:
                lift_range_right.append(i.get('lift_range_right'))
                request.session['lift_range_right'] = lift_range_right

            for i in table:
                lift_range_left.append(i.get('lift_range_left'))
                request.session['lift_range_left'] = lift_range_left

            data = table
            request.session['data'] = data
            print('DATA:', data)

            import matplotlib.pyplot as plt
            import numpy as np

            list_of_values = list_of_values_right
            print(list_of_values)
            values =np.array(list_of_values)
            myvalues = [0, 1]
            labels = ['Non-Defective', 'Defective']

            sum_of_true = [x for x in values if x == 1]
            sum_of_false = [x for x in values if x == 0]

            addition_of_true = len(sum_of_true)
            addition_of_false = len(sum_of_false)

            fig1 = go.Figure(data=[go.Pie(labels=labels, values=[addition_of_true, addition_of_false], pull=[0.1, 0.1])])
            # fig1.update_layout(margin=dict(t=2, b=2, l=2, r=2))
            fig1.update_layout(
                autosize=False,
                width=445,
                height=450
            )
            fig1.update_traces(marker=dict(colors=['#12ABDB', '#0070AD']))
            plot_div = offline.plot(fig1, output_type='div')

            total = round(addition_of_true + addition_of_false)
            y = np.array([addition_of_true, addition_of_false])
            mylabels = [str(round(addition_of_true)), str(round(addition_of_false))]
            s = ['Good', 'Bad']
            good = round(addition_of_true)
            bad = round(addition_of_false)
            plt_1 = plt.figure(figsize=(5, 5))
            colors = ['#12ABDB', '#0070AD']
            plt.pie(y, labels=mylabels, colors=colors)
            plt.legend(['Non-Defective', 'Defective'])
            plt.show()

            plt_1.savefig("media/piechart/capsule_pie")

            import pandas as pd
            # x = round(list_of_values, 5)

            print('list_of_values:', list_of_values)
            print('list_of_files:', list_of_files)


            df = pd.DataFrame({
                'list_of_values': list_of_values,
                'Image': list_of_values,
                'lower_bound_right': lower_bound_right,
                'list_of_values_left': list_of_values_left,
                'lift_range_left': lift_range_left,

            })

            col_left = []
            for x in list_of_values:
                if x == 1:
                    col_left.append('#12ABDB')
                else:
                    col_left.append('#0070AD')

            fig = plot([Bar(x=list_of_files, y=list_of_values_left, marker={'color': col_left},
                            name='test',
                            opacity=0.8, )],
                       output_type='div', image_height=30, image_width=340,)

            range1_list = [x for x in list_of_values if x <= 1]
            range2_list = [x for x in list_of_values if x > 1]
            df.plot(kind='bar', y='lift_range_left', figsize=(7, 7), color=col_left)
            plt.xlabel("Input Image", fontsize=14)
            plt.ylabel("Lift Range Left", fontsize=14)
            # df.plot(kind='bar', x=range2_list, y='Image', figsize=(7, 7), color='red')
            plt.subplots_adjust(bottom=0.2)
            plt.savefig("media/piechart/capsule_bar")

            col_right = []
            for x in list_of_values_right:
                if x == 1:
                    col_right.append('#12ABDB')
                else:
                    col_right.append('#FF6327')

            fig2 = plot([Bar(x=list_of_files, y=list_of_values_right, marker={'color': col_right},
                            name='test',
                            opacity=0.8, )],
                       output_type='div', image_height=50, image_width=300,)

            df = pd.DataFrame({
                'list_of_values': list_of_values,
                'Image': list_of_values,
                'lower_bound_right': lower_bound_right,
                'list_of_values_right': list_of_values_right,
                'lift_range_right': lift_range_right,

            })

            range1_list = [x for x in list_of_values if x <= 1]
            range2_list = [x for x in list_of_values if x > 1]
            df.plot(kind='bar', y='lift_range_right', figsize=(7, 7), color=col_right)
            plt.xlabel("Input Image", fontsize=14)
            plt.ylabel("Lift Range Right", fontsize=14)
            # df.plot(kind='bar', x=range2_list, y='Image', figsize=(7, 7), color='red')
            plt.subplots_adjust(bottom=0.2)
            plt.savefig("media/piechart/capsule_bar2")

            context['lift_iqr_left'] = f"{cpl_df['lift_iqr_left'][0]}"
            context['lift_iqr_right'] = f"{cpl_df['lift_iqr_right'][0]}"

            context = {
                'data': data,
                'images': list_of_files,
                'total': total,
                'good': good,
                'bad': bad,
                'lower_bound_left': request.session.get('lower_bound_left'),
                'upper_bound_left': request.session.get('upper_bound_left'),
                'lower_bound_right': request.session.get('lower_bound_right'),
                'upper_bound_right': request.session.get('upper_bound_right'),
                'plot_div': plot_div,
                'fig': fig,
                'fig2': fig2,
            }
        return render(request, 'result_capsule.html', context)
    return render(request, 'upload_capsule.html')


def result_capsule(request):
    if request.method == 'POST':
        list_of_files = os.listdir('media/capsule_fig1/')
        data = request.session.get('data')
        print('data', data)
        print(list_of_files)
        input_image = request.POST.get('input_image')
        print(input_image)
        context = {
            'data': request.session.get('data'),
            'image': list_of_files,
            'lower_bound_left': request.session.get('lower_bound_left'),
            'upper_bound_left': request.session.get('upper_bound_left'),
            'lower_bound_right': request.session.get('lower_bound_right'),
            'upper_bound_right': request.session.get('upper_bound_right'),
        }
        return render(request, 'detail.html', context)
    return render(request, 'result_capsule.html')



@login_required
def detail_capsule(request):
    number = request.GET.get('search')
    context = {
        'number': number,
        'lower_bound_left': request.session.get('lower_bound_left'),
        'upper_bound_left': request.session.get('upper_bound_left'),
        'lower_bound_right': request.session.get('lower_bound_right'),
        'upper_bound_right': request.session.get('upper_bound_right'),
    }
    return render(request, 'detail_capsule.html', context)
