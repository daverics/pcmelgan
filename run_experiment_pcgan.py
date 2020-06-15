import dataset
import librosa
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as F
from utils import *
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from pathlib import Path
import pandas as pd
import time
from networks import *
# from models import *
# from filter import *
from torch.autograd import Variable
import glob
from mel2wav.modules import MelGAN_Generator, Audio2Mel

LongTensor = torch.cuda.LongTensor
FloatTensor = torch.cuda.FloatTensor


def parse_args():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--device", type = str, default = '0')
    parser.add_argument("--experiment_name", type = str, required = True)
    parser.add_argument("--epochs", type = int, default = 2)
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--save_interval", type = int, default = 1)
    parser.add_argument("--checkpoint_interval", type = int, default = 1)
    parser.add_argument("--load_path", type = str, default = None)
    parser.add_argument("--resume_experiment", type = bool, default = False)
    parser.add_argument("--D_real_loss_weight", type = float, default = 1)
    parser.add_argument("--FD_lr", type = float, default = 4e-4)
    parser.add_argument("--F_lr", type = float, default = 1e-4)
    parser.add_argument("--G_lr", type = float, default = 1e-4)
    parser.add_argument("--GD_lr", type = float, default = 4e-4)
    parser.add_argument("--utility_loss", type = bool, default = False)

    # Model and loss parameters
    parser.add_argument("--loss", type = str, default = None)
    parser.add_argument("--eps", type = float, default = 1e-3)
    parser.add_argument("--lamb", type = float, default = 100)
    parser.add_argument("--entropy_loss", type = bool, default = False)
    parser.add_argument("--filter_receptive_field", type = int, default = 3)
    parser.add_argument("--n_mel_channels", type = int, default = 80)
    parser.add_argument("--ngf", type = int, default = 32)
    parser.add_argument("--n_residual_layers", type = int, default = 3)

    # Data parameters
    parser.add_argument("--sampling_rate", type = int, default = 8000)
    parser.add_argument("--segment_length", type = int, default = 8192)
    parser.add_argument("--data_path", type = str, default = '/home/edvinli/thesis_spring_2020/audio_mnist/')
    parser.add_argument("--meta_data_file", type = str, default = '/home/edvinli/thesis_spring_2020/audio_mnist/audioMNIST_meta.json')

    # Experiment parameters
    parser.add_argument("--seeds", type = int, nargs = '+', default = None)
    parser.add_argument("--num_runs", type = int, default = 3)
    parser.add_argument("--n_completed_runs", type = int, default = None)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    root = Path(os.getcwd())
    experiment_name = args.experiment_name
    dist_loss = args.loss
    num_runs = args.num_runs
    device = 'cuda:' + str(args.device)
    log_dir = os.path.join(root,'logs')
    experiment_dir = os.path.join(log_dir, experiment_name)
    if os.path.exists(experiment_dir) and not args.resume_experiment:
        print("Experiment with this name already exists, use --resume_experiment to continue.")
        exit()

    os.mkdir(experiment_dir)

    # Some hyper parameters
    num_genders = 2
    num_digits = 10
    lamb = args.lamb
    eps = args.eps

    # If manual seed, check the number is same as number of runs
    if not args.seeds == None and len(args.seeds) != num_runs:
            print("Number of provided seeds is not the same as number of training runs")

    # Meta data and list of data files
    annotation_file = args.meta_data_file
    train_file_index = librosa.util.find_files(args.data_path)
    # annotation_file = '/home/edvinli/thesis_spring_2020/audio_mnist/audioMNIST_meta.json'
    # train_file_index = librosa.util.find_files('/home/edvinli/thesis_spring_2020/audio_mnist/')

    split_ratio = 5

    # Build indices for the data
    file_index, annotation_index_gender, annotation_index_digit, annotation_index_speaker_id = dataset.build_annotation_index(
                                                                    train_file_index, annotation_file, balanced_genders = False)
    test_annotation_index, train_annotation_index, test_ids, train_ids = dataset.balanced_annotation_split(file_index, annotation_index_gender, annotation_index_digit, annotation_index_speaker_id, split_ratio)


    # Create the dataset
    train_data = dataset.AnnotatedAudioDataset(
        train_annotation_index, args.sampling_rate, args.segment_length
    )
    test_data = dataset.AnnotatedAudioDataset(
        test_annotation_index, args.sampling_rate, args.segment_length
    )
    n_train = train_data.__len__()
    n_test = test_data.__len__()


    # Dataloaders
    train_loader = DataLoader(train_data, batch_size = args.batch_size, num_workers = 1, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = 1, num_workers = 1)

    # Set up models that are not trained
    fft = Audio2Mel(sampling_rate = args.sampling_rate)
    Mel2Audio = MelGAN_Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).to(device)
    fix_digit_spec_classfier = load_modified_AlexNet(num_digits).to(device)
    fix_gender_spec_classfier = load_modified_AlexNet(num_genders).to(device)

    # Pretrained Mel spectrogram inversion and digit classification
    Mel2Audio.load_state_dict(torch.load('mel2wav/best_netG_epoch_2120.pt'))
    fix_digit_spec_classfier.load_state_dict(torch.load('fixed_classifier_checkpoints/best_digit_alexnet_spectrograms_epoch_26.pt'))
    fix_gender_spec_classfier.load_state_dict(torch.load('fixed_classifier_checkpoints/best_gender_alexnet_epoch_29.pt'))
    fix_digit_spec_classfier.eval()
    fix_gender_spec_classfier.eval()

    # Loss functions
    dist_loss = 'l1'
    distortion_loss = nn.L1Loss()
    entropy_loss = HLoss()
    adversarial_loss = nn.CrossEntropyLoss()
    adversarial_loss_rf = nn.CrossEntropyLoss()

    for run in range(num_runs):
        run_dir = os.path.join(experiment_dir,'run_' + str(run))
        checkpoint_dir = os.path.join(run_dir,'checkpoints')
        visuals_dir = os.path.join(run_dir,'visuals')
        example_dir = os.path.join(run_dir,'examples')
        example_audio_dir = os.path.join(example_dir, 'audio')
        example_spec_dir = os.path.join(example_dir, 'spectrograms')

        if not args.resume_experiment:
            os.mkdir(run_dir)
            os.mkdir(example_dir)
            os.mkdir(checkpoint_dir)
            os.mkdir(example_audio_dir)
            os.mkdir(example_spec_dir)
            os.mkdir(visuals_dir)

        # Set random seed
        if args.seeds == None:
            manualSeed = random.randint(1, 10000) # use if you want new results
        else:
            manualSeed = args.seed[str(run)]
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        np.random.seed(manualSeed)

        ####################################
        # Dump arguments and create logger #
        ####################################
        with open(Path(run_dir) / "args.yml", "w") as f:
            yaml.dump(args, f)
            yaml.dump({'Seed used' : manualSeed}, f)
            yaml.dump({'Run number' : run}, f)
        writer = SummaryWriter(str(run_dir))

        # Set up trainable models and optimizers
        netF = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128], kernel_size = args.filter_receptive_field, image_width=32, image_height=80, noise_dim=10, nb_classes=2, embedding_dim=16, use_cond = False).to(device)
        netFD = load_modified_AlexNet(num_genders).to(device)
        netG = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128], kernel_size = args.filter_receptive_field, image_width=32, image_height=80, noise_dim=10, nb_classes=2, embedding_dim=16, use_cond = True).to(device)
        netGD = load_modified_AlexNet(num_genders + 1).to(device)

        # Optimizers
        optF = torch.optim.Adam(netF.parameters(), args.F_lr, betas = (0.5, 0.9))
        optFD = torch.optim.Adam(netFD.parameters(), args.FD_lr, betas = (0.5, 0.9))
        optG = torch.optim.Adam(netG.parameters(), args.G_lr, betas = (0.5, 0.9))
        optGD = torch.optim.Adam(netGD.parameters(), args.GD_lr, betas = (0.5, 0.9))

        # Put training objects into list for loading and saving state dicts
        training_objects = []
        training_objects.append(('netF', netF))
        training_objects.append(('optF', optF))
        training_objects.append(('netFD', netFD))
        training_objects.append(('optFD', optFD))
        training_objects.append(('netG', netG))
        training_objects.append(('optG', optG))
        training_objects.append(('netGD', netGD))
        training_objects.append(('optGD', optGD))
        training_objects.sort(key = lambda x : x[0])

        # Load from checkpoints
        start_epoch = 0
        if args.resume_experiment or not args.load_path == None:
            if args.resume_experiment:
                if args.n_completed_runs <= run:
                    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*latest*')))
                    start_epoch = int(checkpoints[0].split('_')[-1][:-3])
                    print('Resuming experiment {} from checkpoint, {} epochs completed.'.format(args.experiment_name, start_epoch))
            else:
                checkpoint_path = os.path.join(args.load_path,'checkpoints')
                checkpoints = sorted(glob.glob(os.path.join(checkpoint_path,'*latest*')))
                completed_epochs = int(checkpoints[0].split('_')[-1][:-3])
                print('Starting from checkpoint in {}, {} epochs completed.'.format(args.load_path, completed_epochs))

            for i, (_, object) in enumerate(training_objects):
                object.load_state_dict(torch.load(checkpoints[i]))

        print("GAN training initiated, {} epochs".format(args.epochs))
        for epoch in range(start_epoch, args.epochs + start_epoch):
            # Add counters for number of correct classifications
            correct_FD = 0
            correct_fake_GD = 0
            correct_real_GD = 0
            correct_gender_fake_GD = 0
            correct_digit = 0
            fixed_correct_gender = 0

            # Add variables to add batch losses to
            F_distortion_loss_accum = 0
            F_adversary_loss_accum = 0
            FD_adversary_loss_accum = 0
            G_distortion_loss_accum = 0
            G_adversary_loss_accum = 0
            GD_real_loss_accum = 0
            GD_fake_loss_accum = 0

            netF.train()
            netFD.train()
            netG.train()
            netGD.train()

            epoch_start = time.time()
            for i, (x, gender, digit, _) in enumerate(train_loader):

                digit = digit.to(device)
                gender = gender.to(device)
                x = torch.unsqueeze(x,1)
                spectrograms = fft(x).detach()
                spectrograms, means, stds = preprocess_spectrograms(spectrograms)
                spectrograms = torch.unsqueeze(spectrograms,1).to(device)

                # -----------------
                #  Train Filter
                # -----------------
                optF.zero_grad()

                z = torch.randn(spectrograms.shape[0], 10).to(device)
                filter_mel = netF(spectrograms,z, gender.long())
                pred_secret = netFD(filter_mel)

                ones = Variable(FloatTensor(gender.shape).fill_(1.0), requires_grad = True).to(device)
                target = ones - gender.float()
                target = target.view(target.size(0))
                filter_distortion_loss = distortion_loss(filter_mel, spectrograms)
                if not args.entropy_loss:
                    filter_adversary_loss = adversarial_loss(pred_secret, target.long())
                else:
                    filter_adversary_loss = entropy_loss(pred_secret)
                netF_loss = filter_adversary_loss + lamb * torch.pow(torch.relu(filter_distortion_loss - eps),2)
                netF_loss.backward()
                optF.step()

                # ------------------------
                # Train Generator (Real/Fake)
                # ------------------------
                optG.zero_grad()

                z1 = torch.randn(spectrograms.shape[0], 10).to(device)
                filter_mel = netF(spectrograms,z1,gender.long())

                z2 = torch.randn(spectrograms.shape[0], 10).to(device)
                gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], spectrograms.shape[0]))).to(device)
                gen_mel = netG(filter_mel, z2, gen_secret)
                pred_secret = netGD(gen_mel)
                pred_digit = fix_digit_spec_classfier(gen_mel)
                fixed_pred_secret = fix_gender_spec_classfier(gen_mel)

                generator_distortion_loss = distortion_loss(gen_mel, spectrograms)
                generator_adversary_loss = adversarial_loss(pred_secret, gen_secret)
                netG_loss = generator_adversary_loss + lamb * torch.pow(torch.relu(generator_distortion_loss - eps),2)
                netG_loss.backward()
                optG.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                optFD.zero_grad()

                pred_secret = netFD(filter_mel.detach())
                netFD_loss = adversarial_loss(pred_secret, gender.long())
                netFD_loss.backward()
                optFD.step()

                # --------------------------------
                #  Train Discriminator (Real/Fake)
                # --------------------------------
                optGD.zero_grad()

                real_pred_secret = netGD(spectrograms)
                fake_pred_secret = netGD(gen_mel.detach())

                fake_secret = Variable(LongTensor(fake_pred_secret.size(0)).fill_(2.0), requires_grad=False).to(device)

                GD_real_loss = adversarial_loss_rf(real_pred_secret, gender.long().to(device)).to(device)
                GD_fake_loss = adversarial_loss_rf(fake_pred_secret, fake_secret).to(device)
                netGD_loss = (GD_real_loss + GD_fake_loss)/2
                netGD_loss.backward()
                optGD.step()

                # ----------------------------------------------
                #   Compute accuracies
                # ----------------------------------------------

                # FD accuracy on original gender
                predicted_gender_FD = torch.argmax(pred_secret, 1)
                correct_FD += (predicted_gender_FD == gender.long()).sum()

                # GD accuracy on original gender in real and generated (fake) data,
                # and sampled gender in generated (fake) data
                predicted_fake_GD = torch.argmax(fake_pred_secret, 1)
                predicted_real_GD = torch.argmax(real_pred_secret,1)

                correct_fake_GD += (predicted_fake_GD == fake_secret).sum()
                correct_real_GD += (predicted_real_GD == gender).sum()
                correct_gender_fake_GD += (predicted_fake_GD == gen_secret).sum()

                # Calculate number of correct classifications for the fixed classifiers on the training set
                predicted_digit = torch.argmax(pred_digit.data, 1)
                correct_digit += (predicted_digit == digit).sum()

                fixed_predicted = torch.argmax(fixed_pred_secret.data, 1)
                fixed_correct_gender += (fixed_predicted == gender.long()).sum()

                # ----------------------------------------------
                #   Record losses
                # ----------------------------------------------

                F_distortion_loss_accum += filter_distortion_loss.item()
                F_adversary_loss_accum += filter_adversary_loss.item()
                FD_adversary_loss_accum += netFD_loss.item()
                G_distortion_loss_accum += generator_distortion_loss.item()
                G_adversary_loss_accum += generator_adversary_loss.item()
                GD_real_loss_accum += GD_real_loss.item()
                GD_fake_loss_accum += GD_fake_loss.item()


            writer.add_scalar("F_distortion_loss", F_distortion_loss_accum/(i+1), epoch + 1)
            writer.add_scalar("F_adversary_loss", F_adversary_loss_accum/(i+1), epoch + 1)
            writer.add_scalar("G_distortion_loss", G_distortion_loss_accum/(i+1), epoch + 1)
            writer.add_scalar("G_adversary_loss", G_adversary_loss_accum/(i+1), epoch + 1)
            writer.add_scalar("FD_adversary_loss", FD_adversary_loss_accum/(i+1), epoch + 1)
            writer.add_scalar("GD_real_loss", GD_real_loss_accum/(i+1), epoch + 1)
            writer.add_scalar("GD_fake_loss", GD_fake_loss_accum/(i+1), epoch + 1)

            # ----------------------------------------------
            #   Record accuracies
            # ----------------------------------------------

            FD_accuracy = 100 * correct_FD / n_train
            GD_accuracy_fake = 100 * correct_fake_GD / n_train
            GD_accuracy_real = 100 * correct_real_GD / n_train
            GD_accuracy_gender_fake = 100 * correct_gender_fake_GD / n_train
            fix_digit_spec_classfier_accuracy = 100 * correct_digit / n_train
            fix_gender_spec_classfier_accuracy = 100 * fixed_correct_gender / n_train

            writer.add_scalar("FD_accuracy", FD_accuracy, epoch + 1)
            writer.add_scalar("GD_accuracy_fake", GD_accuracy_fake, epoch + 1)
            writer.add_scalar("GD_accuracy_real", GD_accuracy_real, epoch + 1)
            writer.add_scalar("GD_accuracy_gender_fake", GD_accuracy_gender_fake, epoch + 1)
            writer.add_scalar("digit_accuracy", fix_digit_spec_classfier_accuracy, epoch + 1)
            writer.add_scalar("fixed_gender_accuracy_fake", fix_gender_spec_classfier_accuracy, epoch + 1)

            print('__________________________________________________________________________')
            print("Epoch {} completed | Time: {:5.2f} s ".format(epoch+1, time.time() - epoch_start))
            print("netF    | Adversarial loss: {:5.5f} | Distortion loss: {:5.5f}".format(F_adversary_loss_accum/(i + 1), F_distortion_loss_accum/(i + 1)))
            print("netFD   | Filtered sample accuracy: {} %".format(FD_accuracy))
            print("netG    | Advsarial loss: {:5.5f} | Distortion loss: {:5.5f}".format(G_adversary_loss_accum/(i + 1), G_distortion_loss_accum/(i + 1)))
            print("netGD   | Real samples: {} % | Fake samples: {} % | Sampled gender accuracy: {} % ".format(
                    GD_accuracy_real, GD_accuracy_fake, GD_accuracy_gender_fake
            ))
            print("Fix Digit accuracy: {} % | Fix gender accuracy: {} %".format(fix_digit_spec_classfier_accuracy, fix_gender_spec_classfier_accuracy))

            # ----------------------------------------------
            #   Compute test accuracy
            # ----------------------------------------------

            if epoch % 10 == 0:
                test_correct_digit = 0
                test_fixed_original_gender = 0
                test_fixed_sampled_gender = 0

                for i, (x, gender, digit, speaker_id) in enumerate(test_loader):
                    x = torch.unsqueeze(x,1)
                    spectrograms = fft(x).detach()
                    spectrograms, means, stds = preprocess_spectrograms(spectrograms)
                    spectrograms = torch.unsqueeze(spectrograms,1).to(device)
                    gender = gender.to(device)
                    digit = digit.to(device)

                    z1 = torch.randn(spectrograms.shape[0], 10).to(device)
                    filter_mel = netF(spectrograms,z1, gender.long())
                    z2 = torch.randn(filter_mel.shape[0], 10).to(device)
                    gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], filter_mel.shape[0]))).to(device)
                    gen_mel = netG(filter_mel,z2,gen_secret)

                    pred_digit = fix_digit_spec_classfier(gen_mel)
                    fixed_pred_secret = fix_gender_spec_classfier(gen_mel)

                    # Calculate utility accuracy
                    predicted = torch.argmax(pred_digit.data,1)
                    test_correct_digit += (predicted == digit).sum()

                    # Calculate gender accuracy for fixed net
                    fixed_predicted = torch.argmax(fixed_pred_secret.data,1)
                    test_fixed_original_gender += (fixed_predicted == gender.long()).sum()
                    test_fixed_sampled_gender += (fixed_predicted == gen_secret).sum()

                test_digit_accuracy = 100*test_correct_digit / n_test
                test_fixed_original_gender_accuracy_fake = 100*test_fixed_original_gender / n_test
                test_fixed_sampled_gender_accuracy_fake = 100*test_fixed_sampled_gender / n_test
                writer.add_scalar("test_set_digit_accuracy", test_digit_accuracy, epoch + 1)
                writer.add_scalar("test_set_fixed_original_gender_accuracy_fake", test_fixed_original_gender_accuracy_fake, epoch + 1)
                writer.add_scalar("test_set_fixed_sampled_gender_accuracy_fake", test_fixed_sampled_gender_accuracy_fake, epoch + 1)

                print('__________________________________________________________________________')
                print("## Test set statistics ##")
                print("Utility | Digit accuracy: {} % | Fixed sampled gender accuracy: {} % | Fixed original gender accuracy: {} % ".format(test_digit_accuracy,test_fixed_sampled_gender_accuracy_fake, test_fixed_original_gender_accuracy_fake))

            # ----------------------------------------------
            #   Save test samples
            # ----------------------------------------------

            if (epoch + 1) % args.save_interval == 0:
                print("Saving audio and spectrogram samples.")
                netF.eval()
                netG.eval()
                for i, (x, gender, digit, speaker_id) in enumerate(test_loader):
                    if i % 50 == 0:
                        x = torch.unsqueeze(x,1)
                        spectrograms = fft(x).detach()
                        spec_original = spectrograms
                        spectrograms, means, stds = preprocess_spectrograms(spectrograms)
                        spectrograms = torch.unsqueeze(spectrograms,1).to(device)
                        gender = gender.to(device)
                        digit = digit.to(device)

                        z1 = torch.randn(spectrograms.shape[0], 10).to(device)
                        filtered = netF(spectrograms,z1,gender.long()).detach()

                        z2 = torch.randn(spectrograms.shape[0], 10).to(device)
                        male = Variable(LongTensor(spectrograms.size(0)).fill_(1.0), requires_grad=False).to(device)
                        female = Variable(LongTensor(spectrograms.size(0)).fill_(0.0), requires_grad=False).to(device)
                        generated_male = netG(filtered, z2, male).detach()
                        generated_female = netG(filtered, z2, female).detach()

                        #Predict digit
                        digit_male = fix_digit_spec_classfier(generated_male)
                        pred_digit_male = torch.argmax(digit_male.data,1)
                        digit_female = fix_digit_spec_classfier(generated_female)
                        pred_digit_female = torch.argmax(digit_female.data,1)

                        #Predict gender
                        gender_male = fix_gender_spec_classfier(generated_male)
                        pred_gender_male = torch.argmax(gender_male.data,1)
                        gender_female = fix_gender_spec_classfier(generated_female)
                        pred_gender_female = torch.argmax(gender_female.data,1)

                        if pred_gender_male == 0:
                            pred_gender_male = 'female'
                        else:
                            pred_gender_male = 'male'

                        if pred_gender_female == 0:
                            pred_gender_female = 'female'
                        else:
                            pred_gender_female = 'male'

                        # Distortions
                        filtered_distortion = distortion_loss(spectrograms,filtered)
                        male_distortion = distortion_loss(spectrograms,generated_male).item()
                        female_distortion = distortion_loss(spectrograms,generated_female).item()
                        sample_distortion = distortion_loss(generated_male, generated_female).item()

                        filtered = torch.squeeze(filtered,1).to(device) * 3 * stds.to(device) + means.to(device)
                        generated_male = torch.squeeze(generated_male,1).to(device) * 3 * stds.to(device) + means.to(device)
                        generated_female = torch.squeeze(generated_female,1).to(device) * 3 * stds.to(device) + means.to(device)
                        spectrograms = spectrograms.to(device) * 3 * stds.to(device) + means.to(device)

                        inverted_filtered = Mel2Audio(filtered).squeeze().detach().cpu()
                        inverted_male = Mel2Audio(generated_male).squeeze().detach().cpu()
                        inverted_female = Mel2Audio(generated_female).squeeze().detach().cpu()

                        f_name_filtered_audio = os.path.join(example_audio_dir, 'speaker_{}_digit_{}_epoch_{}_filtered.wav'.format(speaker_id.item(), digit.item(), epoch + 1))
                        f_name_male_audio = os.path.join(example_audio_dir, 'speaker_{}_digit_{}_epoch_{}_sampled_gender_male_predicted_digit_{}.wav'.format(speaker_id.item(), digit.item(), epoch + 1, pred_digit_male.item()))
                        f_name_female_audio =  os.path.join(example_audio_dir, 'speaker_{}_digit_{}_epoch_{}_sampled_gender_female_predicted_digit_{}.wav'.format(speaker_id.item(), digit.item(), epoch + 1, pred_digit_female.item()))
                        f_name_original_audio =  os.path.join(example_audio_dir, 'speaker_{}_digit_{}_.wav'.format(speaker_id.item(), digit.item()))
                        save_sample(f_name_filtered_audio, args.sampling_rate, inverted_filtered)
                        save_sample(f_name_male_audio, args.sampling_rate, inverted_male)
                        save_sample(f_name_female_audio, args.sampling_rate, inverted_female)
                        save_sample(f_name_original_audio,args.sampling_rate,torch.squeeze(x))

                        if gender == 0:
                            gender_title = 'female'
                        else:
                            gender_title = 'male'
                        orig_title = 'Original spectrogram - Gender: {} - Digit: {}'.format(gender_title, digit.item())
                        filtered_title = 'Filtered spectrogram'
                        male_title = 'Sampled/predicted gender: male / {} | Predicted digit: {} \n Distortion loss: {:5.5f} (original) | {:5.5f} (female) ({}_loss)'.format(pred_gender_male, pred_digit_male.item(), male_distortion, sample_distortion, dist_loss)
                        female_title = 'Sampled/predicted gender: female / {} | Predicted digit: {} \n Distortion loss: {:5.5f} (original) | {:5.5f} (male) ({}_loss)'.format(pred_gender_female,pred_digit_female.item(), female_distortion, sample_distortion, dist_loss)
                        f_name = os.path.join(example_spec_dir, 'speaker_{}_digit_{}_epoch_{}'.format(
                                            speaker_id.item(), digit.item(), epoch + 1
                        ))
                        comparison_plot_pcgan(f_name, spec_original, filtered, generated_male, generated_female, orig_title, filtered_title, male_title, female_title)
                print("Success!")



            if (epoch + 1) % args.checkpoint_interval == 0:
                save_epoch = epoch + 1
                old_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*latest*')))
                if old_checkpoints:
                    for i, _ in enumerate(old_checkpoints):
                        os.remove(old_checkpoints[i])
                for name, object in training_objects:
                    torch.save(object.state_dict(), os.path.join(checkpoint_dir, name + '_epoch_{}.pt'.format(save_epoch)))
                    torch.save(object.state_dict(), os.path.join(checkpoint_dir, name + '_latest_epoch_{}.pt'.format(save_epoch)))

        print("Run number {} completed.".format(run+1))
        print('__________________________________________________________________________')

if __name__ == "__main__":
    main()
