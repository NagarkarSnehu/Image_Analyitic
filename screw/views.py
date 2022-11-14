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


# postprocessing.py
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
    with open(f"{os.getcwd()}\\screw/model/info.json", "r") as read_file:
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
        #curr_np_arr = np.repeat(curr_np_arr[..., np.newaxis], 1, -1)
        curr_np_arr = curr_np_arr.astype(np.float32)
        curr_np_arr /= 255.
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
    #curr_np_arr = np.repeat(curr_np_arr[..., np.newaxis], 1, -1)
    curr_np_arr = curr_np_arr.astype(np.float32)
    curr_np_arr /= 255.
    tensor_dict[fname] = curr_np_arr
    print(dir_name, curr_np_arr.shape)
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


def create_in_pred_res_dict(model, images_dict):
    dict_tuples = {}
    for k, imgs in images_dict.items():
        curr_pred = model.predict(imgs)
        print(curr_pred.shape)
        print(imgs.shape)
        #imgs1=np.squeeze(imgs)
        curr_pred1 = np.squeeze(curr_pred,axis=3)
        print(curr_pred1.shape)
        curr_resmap = calculate_resmaps(imgs, curr_pred1, method="ssim")[1]

        dict_tuples[k] = (imgs, curr_pred1, curr_resmap)
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


def screw_prediction(screw_model,
                     data_samples: np.array,
                     data_samples_input: np.array,
                     data_labels_input: list,
                     data_labels: list,
                     screw_data_dict: dict,
                     screw_data_dict_input: dict,
                     Z_FACTOR: float = 1.96, # 95 %CI, use 2.57 for 99% CI
                     ):

    #limit = min(limit, data_samples_input.shape[0])
    #print(limit)
    HISTOGRAM_BINS = 25
    VALS_FILTER = 0.07
    Z_FACTOR = 1.96
    COLOR_MAP = "plasma" # plasma, inferno, cviridis
    MODEL = screw_model
    curr_imgs = data_samples_input

    if len(data_labels_input) == 0:
        data_labels_input = ["TEST"] * curr_imgs.shape[0]
    labels = data_labels_input
    assert curr_imgs.shape[0] == len(labels), "data points and labels should match"
    generated_images = MODEL.predict(curr_imgs)
    generated_images = np.squeeze(generated_images,axis=3)
    _, residual_maps = calculate_resmaps(curr_imgs, generated_images, method="ssim")
    #curr_imgs, generated_images, residual_maps = curr_imgs[:limit], generated_images[:limit], residual_maps[:limit]

    idx = 0
    temp_tbl = {"img":[], # input/good

                "vals_count":[],
                "mean":[],
                "std":[],
                "IQR":[],
                "min":[],
                "25":[],
                "50":[],
                "75":[],
                "max":[],

                }

    final_stat_dict_screw = {
        # "random_samples_idx": [],
        "lower_bound_iqr":[],
        "upper_bound_iqr": [],
        "input_img_iqr":[],
        "lower_bound_range":[],
        "upper_bound_range": [],
        "input_img_range":[],
        "lift_iqr":[],
        "actual_lbl": [],
        "prediction": [],
        "prediction_result": []
    }
    plots = []

    # for COLOR_MAP in ["Greys", "plasma", "Spectral", "seismic", "CMRmap"]:

    for input_img, gen_img, resmap, lbl in zip(curr_imgs, generated_images, residual_maps, labels):
        idx += 1
        curr_temp_tbl = copy.deepcopy(temp_tbl)
        fig1, ax = plt.subplots(1, 3, sharey=True)
        #input_img = np.squeeze(input_img)
        ax[0].imshow(input_img)
        ax[0].title.set_text("input("+lbl+")")
        #gen_img=np.squeeze(gen_img)
        ax[1].imshow(gen_img)

        ax[1].title.set_text("generated")

        ax[2].imshow(resmap, cmap=COLOR_MAP)
        ax[2].title.set_text("resmap")

        fig1.savefig("media/screw_fig1/my_resmap_"+lbl)
        #fig1.show()

        fig2, ax2 = plt.subplots(1, 2, figsize=(9, 3))

        in_resmap = resmap.flatten()
        in_vals = in_resmap[in_resmap > VALS_FILTER]
        in_vector = np.histogram(in_vals, bins=HISTOGRAM_BINS, range=(0, 1))
        curr_temp_tbl["img"].append("input")
        curr_temp_tbl["vals_count"].append(len(in_vals))
        curr_temp_tbl["mean"].append(in_vals.mean())
        curr_temp_tbl["std"].append(in_vals.std(ddof=1))
        curr_temp_tbl["IQR"].append(iqr(in_vals))
        curr_temp_tbl["min"].append(in_vals.min())
        curr_temp_tbl["25"].append(np.quantile(in_vals, 0.25))
        curr_temp_tbl["50"].append(np.quantile(in_vals, 0.5))
        curr_temp_tbl["75"].append(np.quantile(in_vals, 0.75))
        curr_temp_tbl["max"].append(in_vals.max())

        #ax2[0][0].imshow(resmap, cmap=COLOR_MAP)
        #ax2[0][0].title.set_text("in_res")
        ax2[0].hist(in_vals, bins=HISTOGRAM_BINS)

        five_random_indexes = np.random.choice(np.arange(screw_data_dict["good"][0].shape[0]), size=5)
        good_examples = screw_data_dict["good"][0][five_random_indexes]
        good_examples_gen = MODEL.predict(good_examples)
        good_examples_gen = np.squeeze(good_examples_gen)
        #good_examples = np.squeeze(good_examples)
        _, good_examples_resmaps = calculate_resmaps(good_examples, good_examples_gen, method="ssim")

        for idx, g_resmap in zip(range(1, 6), good_examples_resmaps):

            vals = g_resmap.flatten()
            vals = vals[vals > VALS_FILTER]

            curr_temp_tbl["img"].append(f"good_{five_random_indexes[idx - 1]}")
            curr_temp_tbl["vals_count"].append(len(vals))
            curr_temp_tbl["mean"].append(vals.mean())
            curr_temp_tbl["std"].append(vals.std(ddof=1))
            curr_temp_tbl["IQR"].append(iqr(vals))
            curr_temp_tbl["min"].append(vals.min())
            curr_temp_tbl["25"].append(np.quantile(vals, 0.25))
            curr_temp_tbl["50"].append(np.quantile(vals, 0.5))
            curr_temp_tbl["75"].append(np.quantile(vals, 0.75))
            curr_temp_tbl["max"].append(vals.max())

            #ax2[0][idx].title.set_text(f"g_res_{idx}")
            #ax2[0][idx].imshow(g_resmap, cmap=COLOR_MAP)
            #ax2[1][idx].hist(vals, bins=HISTOGRAM_BINS)

            pyplot.hist(vals, bins=HISTOGRAM_BINS,alpha=0.5, density = False,histtype ='step', label=f"good_res_{idx}")

        pyplot.legend(loc='upper right')
        pyplot.show()

        fig2.savefig("media/screw_fig2/my_graph_"+lbl)

        # AVERAGE OF LAST 5
        curr_temp_tbl["img"].append("avg")
        curr_temp_tbl["vals_count"].append(np.mean(curr_temp_tbl["vals_count"][-5:]))
        curr_temp_tbl["mean"].append(np.mean(curr_temp_tbl["mean"][-5:]))
        curr_temp_tbl["std"].append(np.mean(curr_temp_tbl["std"][-5:]))
        curr_temp_tbl["IQR"].append(np.mean(curr_temp_tbl["IQR"][-5:]))
        curr_temp_tbl["min"].append(np.mean(curr_temp_tbl["min"][-5:]))
        curr_temp_tbl["25"].append(np.mean(curr_temp_tbl["25"][-5:]))
        curr_temp_tbl["50"].append(np.mean(curr_temp_tbl["50"][-5:]))
        curr_temp_tbl["75"].append(np.mean(curr_temp_tbl["75"][-5:]))
        curr_temp_tbl["max"].append(np.mean(curr_temp_tbl["max"][-5:]))


        comparison_df = pd.DataFrame(curr_temp_tbl)
        comparison_df["range"] = comparison_df["max"] - comparison_df["min"]
        #display(comparison_df)
        print("-----------------------------------------------------")
        #display(comparison_df.loc[[0, 6], ["img", "IQR", "range"]])

        good_stats = comparison_df.query("img == 'avg'")[["IQR"]]

        iqr_mean = good_stats.loc[:, "IQR"].values[0]
        iqr_std = np.std(curr_temp_tbl["IQR"][-6:-1])
        marginal_err = Z_FACTOR * iqr_std / 4
        iqr_lower_bound, iqr_upper_bound = iqr_mean - marginal_err, iqr_mean + marginal_err
        #print(f"99% CI for IQR - [{iqr_lower_bound}, {iqr_upper_bound}]")

        # Min - max + 99% CI
        range_mean = comparison_df.loc[1:5, "range"].mean()
        range_std = comparison_df.loc[1:5, "range"].std()
        marginal_err_range = Z_FACTOR * range_std / 4
        range_lower, range_upper = 0.82, 0.85
        #print(f"99% CI for RANGE - [{range_lower}, {range_upper}]")

        final_stat_dict_screw["lower_bound_iqr"].append(iqr_lower_bound)
        final_stat_dict_screw["upper_bound_iqr"].append(iqr_upper_bound)
        final_stat_dict_screw["input_img_iqr"].append(comparison_df.loc[0, 'IQR'])
        bound_list=[iqr_lower_bound,iqr_upper_bound]
        closest = min(bound_list, key=lambda x: abs(x - comparison_df.loc[0, 'IQR']))
        #print(closest,bound_list)
        if (iqr_lower_bound<=comparison_df.loc[0, 'IQR']) & (iqr_upper_bound>=comparison_df.loc[0, 'IQR']):
            lift=1
        else:
            lift=abs(comparison_df.loc[0, 'IQR']-closest)+1
        final_stat_dict_screw["lift_iqr"].append(lift)
        final_stat_dict_screw["actual_lbl"].append(lbl)
        final_stat_dict_screw["lower_bound_range"].append(0.82)
        final_stat_dict_screw["upper_bound_range"].append(0.85)
        final_stat_dict_screw["input_img_range"].append(comparison_df.loc[0, 'range'])

        if (range_lower <= comparison_df.loc[0, 'range']) & (range_upper >= comparison_df.loc[0, 'range']):
            final_stat_dict_screw['prediction'].append('non-defective')
            final_stat_dict_screw['prediction_result'].append(1)
        else:
            final_stat_dict_screw['prediction'].append('defective')
            final_stat_dict_screw['prediction_result'].append(0)

    final_stat_screw_df = pd.DataFrame(final_stat_dict_screw)
    final_stat_screw_df["iqr_within_CI"] = (
            (final_stat_screw_df["lower_bound_iqr"] <= final_stat_screw_df["input_img_iqr"])
            & (final_stat_screw_df["input_img_iqr"] <= final_stat_screw_df["upper_bound_iqr"])
    )


    final_stat_screw_df["good"] = final_stat_screw_df["iqr_within_CI"]

    return final_stat_screw_df, plots


model_path = f"{os.getcwd()}\\screw/model/mvtecCAE_b8_e3.hdf5"
screw_model, info, _ = load_model_HDF5(model_path)
screw_model.summary()

from django.conf import settings


def upload_screw(request):
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

            screw_imgs = get_img_dict(Path(r"./screw/data"))
            screw_data_dict = create_in_pred_res_dict(screw_model, screw_imgs)
            all_merged_screw, all_merged_screw_label = create_images_batch_from_dict(screw_data_dict)
            for idx, k in enumerate(screw_data_dict.keys()):
                f, h, r = screw_data_dict[k]
                print(f"{idx, k}\norig - {f.shape}")
                print(f"gene - {h.shape}")
                print(f"resi - {r.shape}")
                print()
        #
        else:
            fs = FileSystemStorage(location='media/screw')
            for i in file_list:
                path = os.listdir(settings.MEDIA_ROOT_TRANSISTOR)
                try:
                    if i.name in path:
                        # path.remove(i.name)
                        os.remove('media/screw/' + i.name)
                except Exception as ex:
                    print(ex)

                fs.save(i.name, i)
                # context["url_{}".format(i)] = fs.url(fs.save(i.name, i))
                # list_of_files = fs.url(fs.save(i.name, i))
                # path = os.path.join(settings.BASE_DIR, 'media\\test\\')

                list_of_files = os.listdir('media/screw')
                context['list_of_files'] = list_of_files

            print(list_of_files)
            screw_imgs = get_img_dict(Path(r"./screw/data"))
            print(screw_imgs.keys())
            screw_data_dict = create_in_pred_res_dict(screw_model, screw_imgs)
            all_merged_screw, all_merged_screw_label = create_images_batch_from_dict(screw_data_dict)
            for idx, k in enumerate(screw_data_dict.keys()):
                f, h, r = screw_data_dict[k]
                print(f"{idx, k}\norig - {f.shape}")
                print(f"gene - {h.shape}")
                print(f"resi - {r.shape}")
                print()

        if assessment_portfolio == '1':
            pass
        else:
            screw_imgs_input = get_img_dict_mutiple(Path(rf"{'media/screw'}"))
            screw_data_dict_input = create_in_pred_res_dict(screw_model, screw_imgs_input)
            all_merged_screw_input, all_merged_screw_label_input = create_images_batch_from_dict(screw_data_dict_input)
            for idx, k in enumerate(screw_data_dict_input.keys()):
                f_input, h_input, r_input = screw_data_dict_input[k]
                print(f"{idx, k}\norig - {f_input.shape}")
                print(f"gene - {h_input.shape}")
                print(f"resi - {r_input.shape}")
                print()

            spl_df, spl_plots = screw_prediction(screw_model, all_merged_screw[0], all_merged_screw_input[0], all_merged_screw_label_input,
                                                      all_merged_screw_label,
                                                      screw_data_dict, screw_data_dict_input,
                                                      Z_FACTOR = 1.96 # 95 %CI, use 2.57 for 99% CI
                                                      )
            print(spl_df)
            table1 = json.loads(spl_df.to_json(orient='records'))
            context['table1'] = table1

            list_of_values = []
            lower_bound = []
            upper_bound = []
            input_img = []
            for i in table1:
                list_of_values.append(i.get('prediction_result'))
                request.session['list_of_values'] = list_of_values

            for i in table1:
                lower_bound.append(i.get('lower_bound_range'))
                request.session['lower_bound'] = lower_bound

            for i in table1:
                upper_bound.append(i.get('upper_bound_range'))
                request.session['upper_bound'] = upper_bound

            for i in table1:
                input_img.append(i.get('input_img_range'))
                request.session['input_img'] = input_img

            print(lower_bound)
            print(upper_bound)
            print(input_img)

            print(list_of_values)
            print(table1)
            print(type(table1))
            data = table1
            request.session['data'] = data
            table_data1 = {}
            table_data1['table1'] = table1
            context['table_data1'] = table_data1
            print(table_data1)
            # context['lift_iqr'] = f"{btl_df['lift_iqr']}"
            context['lift_iqr'] = (spl_df.to_dict()["lift_iqr"])
            print(list_of_files)
            import matplotlib.pyplot as plt
            import numpy as np

            values =np.array(list_of_values)
            labels = ['True', 'False']

            sum_of_true = [x for x in values if x == 1]
            sum_of_false = [x for x in values if x == 0]

            addition_of_true = len(sum_of_true)
            addition_of_false = len(sum_of_false)

            total = round(addition_of_true + addition_of_false)
            y = np.array([addition_of_true, addition_of_false])
            mylabels = [str(round(addition_of_true)), str(round(addition_of_false))]
            s = ['Good', 'Bad']
            good = round(addition_of_true)
            bad = round(addition_of_false)
            plt_1 = plt.figure(figsize=(5, 5))
            colors = ['green', 'red']
            plt.pie(y, labels=mylabels, colors=colors)
            plt.legend(['Non-Defective', 'Defective'])
            plt.show()

            plt_1.savefig("media/piechart/screw_pie4")

            import pandas as pd
            # x = round(list_of_values, 5)

            print('list_of_values:', list_of_values)
            print('list_of_files:', list_of_files)
            df = pd.DataFrame({
                'list_of_values': list_of_values,
                'Image': list_of_values,
                'lower_bound': lower_bound,
                'input_img': input_img,
            })

            range1_list = [x for x in list_of_values if x <= 1]
            range2_list = [x for x in list_of_values if x > 1]
            df.plot(kind='line', y='input_img', figsize=(7, 7), color='green')
            # df.plot(kind='bar', x=range2_list, y='Image', figsize=(7, 7), color='red')
            plt.subplots_adjust(bottom=0.2)
            plt.savefig("media/piechart/screw_bar3")

            print(data)

            context = {
                'data': request.session.get('data'),
                'images': list_of_files,
                'lower_bound': request.session.get('lower_bound'),
                'upper_bound': request.session.get('upper_bound'),
                'input_img': request.session.get('input_img'),
                'list_of_values': request.session.get('list_of_values'),
                'total': total,
                'good': good,
                'bad': bad,
                # 'data1': data1,
            }
            return render(request, 'result_screw.html', context)
    return render(request, 'upload_screw.html')


@login_required
def result_screw(request):
    global images
    if request.method == 'POST':
        # import ipdb
        # ipdb.set_trace()
        list_of_files = os.listdir('media/screw_fig1/')
        data = request.session.get('data')
        print('data', data)
        print(list_of_files)
        input_image = request.POST.get('input_image')
        print(input_image)
        context = {
            'data': request.session.get('data'),
            'image': list_of_files,
            'lower_bound': request.session.get('lower_bound'),
            'upper_bound': request.session.get('upper_bound'),
            'input_img': request.session.get('input_img'),
            'list_of_values': request.session.get('list_of_values'),
        }
        return render(request, 'detail_screw.html', context)
    return render(request, 'result_screw.html')


@login_required
def detail_screw(request):
    number = request.GET.get('search')
    context = {
        'number': number,
        'lower_bound': request.session.get('lower_bound'),
        'upper_bound': request.session.get('upper_bound'),
        'input_img': request.session.get('input_img'),
        'list_of_values': request.session.get('list_of_values'),
    }
    return render(request, 'detail_screw.html', context)
