import dataset
import librosa
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as F
from utils import *
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from pathlib import Path
import pandas as pd
import time
#from models import *
#from filter import *
from networks import *
from torch.autograd import Variable
import glob
from mel2wav.modules import MelGAN_Generator, Audio2Mel

LongTensor = torch.cuda.LongTensor
FloatTensor = torch.cuda.FloatTensor

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type = str, default = '0')
    parser.add_argument("--experiment_path", type = str, default = None)
    parser.add_argument("--save_specs", type = bool, default = False)
    parser.add_argument("--save_audio", type = bool, default = False)

    parser.add_argument("--filter_receptive_field", type = int, default = 3)

    parser.add_argument("--n_mel_channels", type = int, default = 80)
    parser.add_argument("--ngf", type = int, default = 32)
    parser.add_argument("--n_residual_layers", type = int, default = 3)

    # Data parameters
    parser.add_argument("--sampling_rate", type = int, default = 8000)
    parser.add_argument("--segment_length", type = int, default = 8192)

    parser.add_argument("--batch_size", type = int, default = 8)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    runs = sorted(listdirs(args.experiment_path))
    num_runs = len(runs)

    # Some hyper parameters
    num_genders = 2
    num_digits = 10
    device = 'cuda:' + args.device
    print(device)

    # Meta data and list of data files
    annotation_file = '/home/edvinli/thesis_spring_2020/audio_mnist/audioMNIST_meta.json'
    train_file_index = librosa.util.find_files('/home/edvinli/thesis_spring_2020/audio_mnist/')

    split_ratio = 5

    # Build indices for the data
    file_index, annotation_index_gender, annotation_index_digit, annotation_index_speaker_id = dataset.build_annotation_index(
                                                                    train_file_index, annotation_file, balanced_genders = False)
    test_annotation_index, train_annotation_index, test_ids, train_ids = dataset.balanced_annotation_split(file_index, annotation_index_gender, annotation_index_digit, annotation_index_speaker_id, split_ratio)

    # Create the dataset
    test_data = dataset.AnnotatedAudioDataset(
        test_annotation_index, args.sampling_rate, args.segment_length
    )
    n_test = test_data.__len__()

    if args.save_audio:
        test_loader = DataLoader(test_data, batch_size = 25, num_workers = 1)
    else:
        test_loader = DataLoader(test_data, batch_size = args.batch_size, num_workers = 1)


    # Set up models that are not trained
    fft = Audio2Mel(sampling_rate = args.sampling_rate)
    Mel2Audio = MelGAN_Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).to(device)
    utility_netD = load_modified_AlexNet(num_digits).to(device)
    fixed_netD = load_modified_AlexNet(num_genders).to(device)
    utility_audio_net = AudioNet(num_digits).to(device)
    privacy_audio_net = AudioNet(num_genders).to(device)

    spec_FID_net = FID_AlexNet(num_digits).to(device)
    audio_FID_net = AudioNet(num_digits).to(device)

    # Pretrained Mel spectrogram inversion and digit classification
    Mel2Audio.load_state_dict(torch.load('mel2wav/best_netG_epoch_2120.pt'))
    utility_netD.load_state_dict(torch.load('fixed_classifier_checkpoints/best_digit_alexnet_spectrograms_epoch_26.pt'))
    fixed_netD.load_state_dict(torch.load('fixed_classifier_checkpoints/best_gender_alexnet_epoch_29.pt'))
    utility_audio_net.load_state_dict(torch.load('fixed_classifier_checkpoints/audio_digit_net_early_stop_epoch_26.pt'))
    privacy_audio_net.load_state_dict(torch.load('fixed_classifier_checkpoints/audio_gender_net_early_stop_epoch_36.pt'))

    utility_netD.eval()
    fixed_netD.eval()
    utility_audio_net.eval()
    privacy_audio_net.eval()

    # Pretrained FID loading
    spec_FID_net.load_state_dict(torch.load('fixed_classifier_checkpoints/best_digit_alexnet_spectrograms_epoch_26.pt'))
    audio_FID_net.load_state_dict(torch.load('fixed_classifier_checkpoints/audio_digit_net_early_stop_epoch_26.pt'))

    # Initialize arrays for accuracies and fid scores
    spec_digit_accuracy_F = []
    spec_original_gender_accuracy_F = []

    spec_digit_accuracy_G = []
    spec_original_gender_accuracy_G = []
    spec_sampled_gender_accuracy_G = []

    audio_digit_accuracy_F = []
    audio_original_gender_accuracy_F = []

    audio_digit_accuracy_G = []
    audio_original_gender_accuracy_G = []
    audio_sampled_gender_accuracy_G = []

    fid_spec_F = []
    fid_spec_G = []
    fid_audio_F = []
    fid_audio_G = []
    fid_inverted_audio = []

    for i in range(num_runs):
        run_path = os.path.join(args.experiment_path, runs[i])
        result_dir = os.path.join(run_path, 'results')
        audio_result_dir = os.path.join(result_dir, 'audio')
        spec_result_dir =  os.path.join(result_dir, 'spectrograms')
        checkpoint_dir = os.path.join(run_path, 'checkpoints')

        # os.mkdir(result_dir)
        # os.mkdir(audio_result_dir)
        # os.mkdir(spec_result_dir)

        # Set up and load trained model
        netF = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128], kernel_size = args.filter_receptive_field, image_width=32, image_height=80, noise_dim=10, nb_classes=2, embedding_dim=16, use_cond = False).to(device)
        netG = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128], kernel_size = args.filter_receptive_field, image_width=32, image_height=80, noise_dim=10, nb_classes=2, embedding_dim=16, use_cond = True).to(device)
        netF.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'netF_latest_epoch_1000.pt')))
        netG.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'netG_latest_epoch_1000.pt')))

        spec_correct_digit_F = 0
        spec_correct_original_gender_F = 0

        spec_correct_digit_G = 0
        spec_correct_original_gender_G = 0
        spec_correct_sampled_gender_G = 0

        audio_correct_digit_F = 0
        audio_correct_original_gender_F = 0
        audio_correct_digit_G = 0
        audio_correct_original_gender_G = 0
        audio_correct_sampled_gender_G = 0

        acts_real_spec = []
        acts_fake_spec_F = []
        acts_fake_spec_G = []

        acts_real_audio = []
        acts_fake_audio_F = []
        acts_fake_audio_G = []
        acts_inverted_audio = []

        for j, (x, gender, digit, speaker_id) in enumerate(test_loader):
            x = torch.unsqueeze(x,1)
            spectrograms = fft(x).detach()
            spectrograms, means, stds = preprocess_spectrograms(spectrograms)
            spectrograms = torch.unsqueeze(spectrograms,1).to(device)
            gender = gender.to(device)
            digit = digit.to(device)

            # --------------------------
            # Spectrogram calculations
            # --------------------------

            z1 = torch.randn(spectrograms.shape[0], 10).to(device)
            filter_mel = netF(spectrograms,z1, gender.long())
            z2 = torch.randn(filter_mel.shape[0], 10).to(device)
            gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], filter_mel.shape[0]))).to(device)
            gen_mel = netG(filter_mel,z2,gen_secret)

            spec_pred_digit_F = utility_netD(filter_mel)
            spec_pred_gender_F = fixed_netD(filter_mel)

            spec_pred_digit_G = utility_netD(gen_mel)
            spec_pred_gender_G = fixed_netD(gen_mel)


            # Calculate digit accuracy for fixed net on F and G outputs
            pred_digit_F = torch.argmax(spec_pred_digit_F.data,1)
            spec_correct_digit_F += (pred_digit_F == digit).sum().item()

            pred_digit_G = torch.argmax(spec_pred_digit_G.data,1)
            spec_correct_digit_G += (pred_digit_G == digit).sum().item()

            # Calculate gender accuracy for fixed net on F and G outputs
            pred_gender_F = torch.argmax(spec_pred_gender_F.data,1)
            spec_correct_original_gender_F += (pred_gender_F == gender.long()).sum().item()

            pred_gender_G = torch.argmax(spec_pred_gender_G.data,1)
            spec_correct_original_gender_G += (pred_gender_G == gender.long()).sum().item()
            spec_correct_sampled_gender_G += (pred_gender_G == gen_secret).sum().item()

            # Calculate FID on transformed spectrograms

            # acts1_tmp = spec_FID_net(spectrograms)
            # acts2_tmp = spec_FID_net(filter_mel)
            # acts3_tmp = spec_FID_net(gen_mel)
            #
            # acts_real_spec.append(np.squeeze(acts1_tmp.detach().cpu().numpy()))
            # acts_fake_spec_F.append(np.squeeze(acts2_tmp.detach().cpu().numpy()))
            # acts_fake_spec_G.append(np.squeeze(acts3_tmp.detach().cpu().numpy()))

            # Save spectrogram samples
            if args.save_specs:
                z1 = torch.randn(spectrograms.shape[0], 10).to(device)
                filtered = netF(spectrograms,z1,gender.long()).detach()

                z2 = torch.randn(spectrograms.shape[0], 10).to(device)
                male = Variable(LongTensor(spectrograms.size(0)).fill_(1.0), requires_grad=False).to(device)
                female = Variable(LongTensor(spectrograms.size(0)).fill_(0.0), requires_grad=False).to(device)
                generated_male = netG(filtered, z2, male).detach()
                generated_female = netG(filtered, z2, female).detach()

                filtered = torch.squeeze(filtered,1).to(device) * 3 * stds.to(device) + means.to(device)
                generated_male = generated_male.to(device) * 3 * stds.to(device) + means.to(device)
                generated_female = generated_female.to(device) * 3 * stds.to(device) + means.to(device)
                spectrograms = spectrograms.to(device) * 3 * stds.to(device) + means.to(device)

                if gender == 0:
                    gender_title = 'female'
                else:
                    gender_title = 'male'
                orig_title = 'Original spectrogram - Gender: {} - Digit: {}'.format(gender_title, digit.item())
                male_title = 'Sampled gender: male '
                female_title = 'Sampled gender: female'
                filtered_title = 'Filtered spectrogram'
                f_name_original = os.path.join(spec_result_dir, 'speaker_{}_digit_{}_original.png'.format(
                                    speaker_id.item(), digit.item()
                ))
                f_name_male = os.path.join(spec_result_dir, 'speaker_{}_digit_{}_male.png'.format(
                                    speaker_id.item(), digit.item()
                ))
                f_name_female = os.path.join(spec_result_dir, 'speaker_{}_digit_{}_female.png'.format(
                                    speaker_id.item(), digit.item()
                ))
                f_name_filtered = os.path.join(spec_result_dir, 'speaker_{}_digit_{}_filtered.png'.format(speaker_id.item(), digit.item()
		))

                save_spec_plot(f_name_original, spectrograms, orig_title)
                save_spec_plot(f_name_male, generated_male, male_title)
                save_spec_plot(f_name_female, generated_female, female_title)
                save_spec_plot(f_name_filtered, filtered, filtered_title)

            # --------------------------
            # Audio calculations
            # --------------------------

            # Denormalize spectrograms before inversion
            filter_mel = torch.squeeze(filter_mel,1).to(device) * 3 * stds.to(device) + means.to(device)
            gen_mel = torch.squeeze(gen_mel,1).to(device) * 3 * stds.to(device) + means.to(device)
            spectrograms = torch.squeeze(spectrograms,1).to(device) * 3 * stds.to(device) + means.to(device)

            # Invert spectrograms using MelGAN
            original_audio = Mel2Audio(spectrograms)
            filter_audio = Mel2Audio(filter_mel)
            gen_audio = Mel2Audio(gen_mel)

            # Classify transformed audio
            audio_pred_digit_F, _ = utility_audio_net(filter_audio)
            audio_pred_gender_F, _ = privacy_audio_net(filter_audio)

            audio_pred_digit_G, _ = utility_audio_net(gen_audio)
            audio_pred_gender_G, _ = privacy_audio_net(gen_audio)

            pred_digit_F = torch.argmax(audio_pred_digit_F.data, 1)
            pred_gender_F = torch.argmax(audio_pred_gender_F.data, 1)

            pred_digit_G = torch.argmax(audio_pred_digit_G.data, 1)
            pred_gender_G = torch.argmax(audio_pred_gender_G.data, 1)

            audio_correct_digit_F += (pred_digit_F == digit).sum().item()
            audio_correct_original_gender_F += (pred_gender_F == gender.long()).sum().item()

            audio_correct_digit_G += (pred_digit_G == digit).sum().item()
            audio_correct_sampled_gender_G += (pred_gender_G == gen_secret).sum().item()
            audio_correct_original_gender_G += (pred_gender_G == gender.long()).sum().item()

            # Compute activations for FID calculations for the transformed audio

            _, acts1_tmp = audio_FID_net(x.to(device))
            _, acts2_tmp = audio_FID_net(filter_audio.to(device))
            _, acts3_tmp = audio_FID_net(gen_audio.to(device))
            _, acts4_tmp = audio_FID_net(original_audio.to(device))

            acts1_tmp = torch.flatten(acts1_tmp,1)
            acts2_tmp = torch.flatten(acts2_tmp,1)
            acts3_tmp = torch.flatten(acts3_tmp,1)
            acts4_tmp = torch.flatten(acts4_tmp,1)

            acts_real_audio.append(np.squeeze(acts1_tmp.detach().cpu().numpy()))
            acts_fake_audio_F.append(np.squeeze(acts2_tmp.detach().cpu().numpy()))
            acts_fake_audio_G.append(np.squeeze(acts3_tmp.detach().cpu().numpy()))
            acts_inverted_audio.append(np.squeeze(acts4_tmp.detach().cpu().numpy()))

            # --------------------------
            # Save audio sample
            # --------------------------
            if args.save_audio:
                if j % 2 == 0:
                    original_audio_sample = torch.squeeze(original_audio[0]).detach().cpu()
                    gen_audio_sample = torch.squeeze(gen_audio[0]).detach().cpu()
                    speaker_id_sample = speaker_id[0].detach().cpu()
                    digit_sample = digit[0].detach().cpu()
                    gender_sample = gender[0].detach().cpu()
                    gen_secret_sample = gen_secret[0].detach().cpu()

                    f_name_orig_audio = os.path.join(audio_result_dir,'speaker_{}_digit_{}_original_inverted.wav'.format(speaker_id_sample, digit_sample, gender_sample, gen_secret_sample))
                    f_name_gen_audio = os.path.join(audio_result_dir,'speaker_{}_digit_{}_gender_orig_{}_sampled_{}.wav'.format(speaker_id_sample, digit_sample, gender_sample, gen_secret_sample))

                    save_sample(f_name_orig_audio, args.sampling_rate, original_audio_sample)
                    save_sample(f_name_gen_audio, args.sampling_rate, gen_audio_sample)

        # Calcuate accuracies
        spec_digit_accuracy_F.append(100*spec_correct_digit_F / n_test)
        spec_original_gender_accuracy_F.append(100*spec_correct_original_gender_F / n_test)

        spec_digit_accuracy_G.append(100*spec_correct_digit_G / n_test)
        spec_original_gender_accuracy_G.append(100*spec_correct_original_gender_G / n_test)
        spec_sampled_gender_accuracy_G.append(100*spec_correct_sampled_gender_G / n_test)

        audio_digit_accuracy_F.append(100*audio_correct_digit_F / n_test)
        audio_original_gender_accuracy_F.append(100*audio_correct_original_gender_F / n_test)

        audio_digit_accuracy_G.append(100*audio_correct_digit_G / n_test)
        audio_original_gender_accuracy_G.append(100*audio_correct_original_gender_G / n_test)
        audio_sampled_gender_accuracy_G.append(100*audio_correct_sampled_gender_G / n_test)

        # Concatenate batch activations into single array
        # acts_real_spec = np.concatenate(acts_real_spec, axis = 0)
        # acts_fake_spec_F = np.concatenate(acts_fake_spec_F, axis = 0)
        # acts_fake_spec_G = np.concatenate(acts_fake_spec_G, axis = 0)

        acts_real_audio = np.concatenate(acts_real_audio, axis = 0)
        acts_fake_audio_F = np.concatenate(acts_fake_audio_F, axis = 0)
        acts_fake_audio_G = np.concatenate(acts_fake_audio_G, axis = 0)
        acts_inverted_audio = np.concatenate(acts_inverted_audio, axis = 0)

        # Calculate FID scores

        # fid_spec_tmp_F = compute_frechet_inception_distance(acts_real_spec, acts_fake_spec_F)
        # fid_spec_tmp_G = compute_frechet_inception_distance(acts_real_spec, acts_fake_spec_G)
        #fid_audio_tmp_F = compute_frechet_inception_distance(acts_real_audio, acts_fake_audio_F)
        #fid_audio_tmp_G = compute_frechet_inception_distance(acts_real_audio, acts_fake_audio_G)
        #if i == 0:
    #       fid_inverted_audio_tmp = compute_frechet_inception_distance(acts_real_audio, acts_inverted_audio)

        # fid_spec_F.append(fid_spec_tmp_F)
        #fid_audio_F.append(fid_audio_tmp_F)
        # fid_spec_G.append(fid_spec_tmp_G)
        #fid_audio_G.append(fid_audio_tmp_G)
        #fid_inverted_audio.append(fid_inverted_audio_tmp)

        print("Computed accuracies and FID for run {}.".format(i))

    # Create data frames with the accuracies
    spec_digit_acc_F_df = pd.DataFrame(spec_digit_accuracy_F)
    spec_orig_gender_acc_F_df = pd.DataFrame(spec_original_gender_accuracy_F)

    spec_digit_acc_G_df = pd.DataFrame(spec_digit_accuracy_G)
    spec_orig_gender_acc_G_df = pd.DataFrame(spec_original_gender_accuracy_G)
    spec_sampled_gender_acc_G_df = pd.DataFrame(spec_sampled_gender_accuracy_G)

    audio_digit_acc_F_df = pd.DataFrame(audio_digit_accuracy_F)
    audio_orig_gender_acc_F_df = pd.DataFrame(audio_original_gender_accuracy_F)

    audio_digit_acc_G_df = pd.DataFrame(audio_digit_accuracy_G)
    audio_orig_gender_acc_G_df = pd.DataFrame(audio_original_gender_accuracy_G)
    audio_sampled_gender_acc_G_df = pd.DataFrame(audio_sampled_gender_accuracy_G)

    fid_spec_F_df = pd.DataFrame(fid_spec_F)
    fid_audio_F_df = pd.DataFrame(fid_audio_F)

    fid_spec_G_df = pd.DataFrame(fid_spec_G)
    fid_audio_G_df = pd.DataFrame(fid_audio_G)
    fid_inverted_audio_df = pd.DataFrame(fid_inverted_audio)

    # Save accuracies and FID scores
    spec_digit_acc_F_df.to_csv(os.path.join(args.experiment_path, 'spec_digit_acc_F.csv'), index = False)
    spec_orig_gender_acc_F_df.to_csv(os.path.join(args.experiment_path, 'spec_orig_gender_acc_F.csv'), index = False)

    spec_digit_acc_G_df.to_csv(os.path.join(args.experiment_path, 'spec_digit_acc_G.csv'), index = False)
    spec_orig_gender_acc_G_df.to_csv(os.path.join(args.experiment_path, 'spec_orig_gender_acc_G.csv'), index = False)
    spec_sampled_gender_acc_G_df.to_csv(os.path.join(args.experiment_path, 'spec_sampled_gender_acc_G.csv'), index = False)

    audio_digit_acc_F_df.to_csv(os.path.join(args.experiment_path, 'audio_digit_acc_F.csv'), index = False)
    audio_orig_gender_acc_F_df.to_csv(os.path.join(args.experiment_path, 'audio_orig_gender_acc_F.csv'), index = False)

    audio_digit_acc_G_df.to_csv(os.path.join(args.experiment_path, 'audio_digit_acc_G.csv'), index = False)
    audio_orig_gender_acc_G_df.to_csv(os.path.join(args.experiment_path, 'audio_orig_gender_acc_G.csv'), index = False)
    audio_sampled_gender_acc_G_df.to_csv(os.path.join(args.experiment_path, 'audio_sampled_gender_acc_G.csv'), index = False)

    fid_spec_F_df.to_csv(os.path.join(args.experiment_path, 'fid_spec_F.csv'), index = False)
    fid_audio_F_df.to_csv(os.path.join(args.experiment_path, 'fid_audio_F.csv'), index = False)

    fid_spec_G_df.to_csv(os.path.join(args.experiment_path, 'fid_spec_G.csv'), index = False)
    fid_audio_G_df.to_csv(os.path.join(args.experiment_path, 'fid_audio_G.csv'), index = False)
    fid_inverted_audio_df.to_csv(os.path.join(args.experiment_path, 'fid_inverted_audio_9_june_G.csv'), index = False)

if __name__ == "__main__":
    main()
