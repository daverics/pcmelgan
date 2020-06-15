import os
import argparse
import tensorflow as tf
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import csv
sns.set()

LongTensor = torch.cuda.LongTensor
FloatTensor = torch.cuda.FloatTensor

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_paths", type = str, nargs = '+', required = True)
    parser.add_argument("--result_dir", type = str, default = '/home/daverics/adversarial_learning_speech/audio_mnist/experiment_results/')
    parser.add_argument("--model", type = str, required = True)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    print(args.experiment_paths)

    save_dir = os.path.join(args.result_dir, args.model)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    audio_digit_acc_F_mean = []
    audio_gender_acc_F_mean = []
    fid_audio_F_mean = []

    audio_digit_acc_F_std = []
    audio_gender_acc_F_std = []
    fid_audio_F_std = []

    audio_digit_acc_G_mean = []
    audio_gender_acc_G_mean = []
    fid_audio_G_mean = []

    audio_digit_acc_G_std = []
    audio_gender_acc_G_std = []
    fid_audio_G_std = []

    for experiment in args.experiment_paths:

        audio_digit_acc_F_list = np.genfromtxt(os.path.join(experiment,'audio_digit_acc_F.csv'),delimiter=',')[1:]
        audio_gender_acc_F_list = np.genfromtxt(os.path.join(experiment,'audio_orig_gender_acc_F.csv'),delimiter=',')[1:]

        audio_digit_acc_G_list = np.genfromtxt(os.path.join(experiment,'audio_digit_acc_G.csv'),delimiter=',')[1:]
        audio_gender_acc_G_list = np.genfromtxt(os.path.join(experiment,'audio_orig_gender_acc_G.csv'),delimiter=',')[1:]

        fid_audio_F_list = np.genfromtxt(os.path.join(experiment,'fid_audio_F.csv'),delimiter=',')[1:]
        fid_audio_G_list = np.genfromtxt(os.path.join(experiment,'fid_audio_G.csv'),delimiter=',')[1:]

        audio_digit_acc_F_mean.append(np.mean(audio_digit_acc_F_list))
        audio_digit_acc_F_std.append(np.std(audio_digit_acc_F_list))
        audio_gender_acc_F_mean.append(np.mean(audio_gender_acc_F_list))
        audio_gender_acc_F_std.append(np.std(audio_gender_acc_F_list))

        audio_digit_acc_G_mean.append(np.mean(audio_digit_acc_G_list))
        audio_digit_acc_G_std.append(np.std(audio_digit_acc_G_list))
        audio_gender_acc_G_mean.append(np.mean(audio_gender_acc_G_list))
        audio_gender_acc_G_std.append(np.std(audio_gender_acc_G_list))

        fid_audio_F_mean.append(np.mean(fid_audio_F_list))
        fid_audio_F_std.append(np.std(fid_audio_F_list))

        fid_audio_G_mean.append(np.mean(fid_audio_G_list))
        fid_audio_G_std.append(np.std(fid_audio_G_list))

    # Generate FID and acc table scores

    with open(os.path.join(save_dir,'fid_audio.csv'), mode='w') as file:
        fid_writer = csv.writer(file, delimiter=',')
        for i in range(4):
            fid_writer.writerow(['$ {:5.2f} \pm {:5.2f} $'.format(fid_audio_F_mean[i], fid_audio_F_std[i]),
            '$ {:5.2f} \pm {:5.2f} $'.format(fid_audio_G_mean[i], fid_audio_G_std[i])

    with open(os.path.join(save_dir,'audio_accs_pcgan.csv'), mode='w') as file:
        fid_writer = csv.writer(file, delimiter=',')
        for i in range(4):
            fid_writer.writerow(['$ {:5.2f} \pm {:5.2f} $'.format(audio_gender_acc_F_mean[i], audio_gender_acc_F_std[i]),
            '$ {:5.2f} \pm {:5.2f} $'.format(audio_gender_acc_G_mean[i], audio_gender_acc_G_std[i]),
            '$ {:5.2f} \pm {:5.2f} $'.format(audio_digit_acc_F_mean[i], audio_digit_acc_F_std[i]),
            '$ {:5.2f} \pm {:5.2f} $'.format(audio_digit_acc_G_mean[i], audio_digit_acc_G_std[i])])


    with open(os.path.join(save_dir,'spec_accs_pcgan.csv'), mode='w') as file:
        fid_writer = csv.writer(file, delimiter=',')
        for i in range(4):
            fid_writer.writerow(['$ {:5.2f} \pm {:5.2f} $'.format(spec_gender_acc_F_mean[i], spec_gender_acc_F_std[i]),
            '$ {:5.2f} \pm {:5.2f} $'.format(spec_gender_acc_G_mean[i], spec_gender_acc_G_std[i]),
            '$ {:5.2f} \pm {:5.2f} $'.format(spec_digit_acc_F_mean[i], spec_digit_acc_F_std[i]),
            '$ {:5.2f} \pm {:5.2f} $'.format(spec_digit_acc_G_mean[i], spec_digit_acc_G_std[i])])


    #label = ['eps=0.01','eps=0.05','eps=0.1','eps=0.2']
    #epsilons = [0.01, 0.05, 0.1, 0.2]
    #labe =['eps=0.1','eps=0.2']
    epsilons = [0.005, 0.01, 0.05, 0.1]

    #Plot digit vs gender accuracy
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.scatter(spec_digit_acc_F_mean, spec_gender_acc_F_mean, c = 'orange', label = 'Filter')
    ax1.scatter(spec_digit_acc_G_mean, spec_gender_acc_G_mean, c = 'blue', label = 'PCMelGAN')
    ax2.scatter(audio_digit_acc_F_mean, audio_gender_acc_F_mean, c = 'orange', label = 'Filter')
    ax2.scatter(audio_digit_acc_G_mean, audio_gender_acc_G_mean, c = 'blue', label = 'PCMelGAN')

    for i, eps in enumerate(epsilons):
        ax1.annotate(r'$\varepsilon$ =' + str(eps) , xy = (spec_digit_acc_F_mean[i], spec_gender_acc_F_mean[i]), xytext = (spec_digit_acc_F_mean[i] - 4, spec_gender_acc_F_mean[i]))
        ax1.annotate(r'$\varepsilon$ =' + str(eps) , xy = (spec_digit_acc_G_mean[i], spec_gender_acc_G_mean[i]), xytext = (spec_digit_acc_G_mean[i] - 4, spec_gender_acc_G_mean[i]))
        ax2.annotate(r'$\varepsilon$ =' + str(eps) , xy = (audio_digit_acc_F_mean[i], audio_gender_acc_F_mean[i]), xytext = (audio_digit_acc_F_mean[i] - 2, audio_gender_acc_F_mean[i]))
        ax2.annotate(r'$\varepsilon$ =' + str(eps) , xy = (audio_digit_acc_G_mean[i], audio_gender_acc_G_mean[i]), xytext = (audio_digit_acc_G_mean[i] - 2, audio_gender_acc_G_mean[i]))

    ax1.set_ylabel('Privacy')
    ax1.set_xlabel('Utility')
    ax2.set_ylabel('Privacy')
    ax2.set_xlabel('Utility')
    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'upper left')
    fig1.savefig(os.path.join(save_dir,'trade_off_plot_spec_15_june.png'))

    ax2.set_ylabel('Privacy')
    ax2.set_xlabel('Utility')
    fig2.savefig(os.path.join(save_dir,'trade_off_plot_audio_15_june.png'))

    plt.close(fig1)
    plt.close(fig2)

    # #Plot Fid vs epsilon
    # #Spectrograms
    # fig, ax = plt.subplots()
    # ax.scatter(epsilons, fid_spec_mean)
    #
    # plt.ylabel('Frechet Inception Distance')
    # plt.xlabel('Epsilon')
    # fig.savefig(os.path.join(save_dir,'fid_spec_plot.png'))
    # plt.close(fig)
    #
    # #Audio
    # fig, ax = plt.subplots()
    # ax.scatter(epsilons, fid_audio_mean)
    #
    # plt.ylabel('Frechet Inception Distance')
    # plt.xlabel('Epsilon')
    # fig.savefig(os.path.join(save_dir,'fid_audio_plot.png'))
    # plt.close(fig)

if __name__ == '__main__':
    main()
