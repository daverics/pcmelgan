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
from models import *
from filter import *
from torch.autograd import Variable
import glob
from mel2wav.modules import MelGAN_Generator

LongTensor = torch.cuda.LongTensor


def parse_args():
    parser = argparse.ArgumentParser()

    # Training parameters
    #parser.add_argument("--device", type = str, required = True)
    parser.add_argument("--experiment_name", type = str, required = True)
    parser.add_argument("--resume_experiment", type = bool, default = False)
    parser.add_argument("--epochs", type = int, default = 10000)
    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--load_path", type = str, default = None)
    parser.add_argument("--stop_interval", type = int, default = 100)

    # Data parameters
    parser.add_argument("--sampling_rate", type = int, default = 8000)
    parser.add_argument("--segment_length", type = int, default = 8192)

    parser.add_argument("--seed", type = int, default = None)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    root = Path(os.getcwd())
    experiment_name = args.experiment_name
    device = 'cuda:0'

    # Set random seed
    if args.seed == None:
        manualSeed = random.randint(1, 10000) # use if you want new results
    else:
        manualSeed = args.seed
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)


    # Set up log directory
    log_dir = os.path.join(root,'logs')
    experiment_dir = os.path.join(log_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir,'checkpoints')
    visuals_dir = os.path.join(experiment_dir,'visuals')
    example_dir = os.path.join(experiment_dir,'examples')
    example_audio_dir = os.path.join(example_dir, 'audio')
    example_spec_dir = os.path.join(example_dir, 'spectrograms')
    if os.path.exists(experiment_dir) and not args.resume_experiment:
        print("Experiment with this name already exists, use --resume_experiment to continue.")
        exit()
    elif not args.resume_experiment:
        os.mkdir(experiment_dir)
        os.mkdir(example_dir)
        os.mkdir(checkpoint_dir)
        os.mkdir(example_audio_dir)
        os.mkdir(example_spec_dir)
        os.mkdir(visuals_dir)

    # ##################################
    # Dump arguments and create logger #
    # ###################################
    with open(Path(experiment_dir) / "args.yml", "w") as f:
        yaml.dump(args, f)
        yaml.dump({'Seed used' : manualSeed}, f)
    writer = SummaryWriter(str(experiment_dir))

    # Some hyper parameters
    num_genders = 2
    num_digits = 10


    # Meta data and list of data files
    annotation_file = '/home/adam/adversarial_learning_speech/audio_mnist/audio_mnist/audioMNIST_meta.json'
    train_file_index = librosa.util.find_files('/home/adam/adversarial_learning_speech/audio_mnist/audio_mnist/')

    split_ratio = 5

    # Build indices for the data
    file_index, annotation_index_gender, annotation_index_digit, annotation_index_speaker_id = dataset.build_annotation_index(
                                                                    train_file_index, annotation_file, balanced_genders = False)
    test_annotation_index, train_annotation_index, test_ids, train_ids = dataset.balanced_annotation_split(file_index, annotation_index_gender, annotation_index_digit, annotation_index_speaker_id, split_ratio)

    print(test_ids)
    print(train_ids)
    # Create the dataset
    train_data = dataset.AnnotatedAudioDataset(train_annotation_index, args.sampling_rate, args.segment_length
    )
    test_data = dataset.AnnotatedAudioDataset(
        test_annotation_index, args.sampling_rate, args.segment_length
    )

    n_train = train_data.__len__()
    n_test = test_data.__len__()

    #Set up models
    audio_gender_net = AudioNet(num_genders).to(device)
    audio_digit_net = AudioNet(num_digits).to(device)

    # Optimizers
    opt_gender = torch.optim.Adam(audio_gender_net.parameters(),1e-4, betas = (0.5, 0.9))
    opt_digit = torch.optim.Adam(audio_digit_net.parameters(), 1e-4, betas = (0.5, 0.9))

    # Put training objects into list for loading and saving state dicts
    training_objects = []
    training_objects.append(('netGender', audio_gender_net))
    training_objects.append(('optGender', opt_gender))
    training_objects.append(('netDigit', audio_digit_net))
    training_objects.append(('optDigit', opt_digit))
    training_objects.sort(key = lambda x : x[0])

    # Loss
    gender_loss = nn.CrossEntropyLoss()
    digit_loss = nn.CrossEntropyLoss()

    lowest_loss_digit = 1e+6
    lowest_loss_gender =1e+6

    counter_digit=0
    counter_gender=0

    # Dataloaders
    train_loader = DataLoader(train_data, batch_size = args.batch_size  , num_workers = 2, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = 1, num_workers = 1)
    iter = 0



    best_test_acc_digit = 0
    best_test_acc_gender = 0
    print("Training initiated, {} epochs".format(args.epochs))
    for epoch in range(0, args.epochs):
        correct_gender = 0
        correct_digit = 0

        epoch_start = time.time()
        for i, (x, gender, digit, _) in enumerate(train_loader):
            audio_digit_net.train()
            audio_gender_net.train()
            x = torch.unsqueeze(x,1).to(device)
            digit = digit.to(device)
            gender = gender.to(device)


            #---------------------
            # Train gender net
            #---------------------

            opt_gender.zero_grad()

            pred_gender, _ = audio_gender_net(x)
            audio_gender_loss = gender_loss(pred_gender, gender)
            audio_gender_loss.backward()
            opt_gender.step()

            #---------------------
            # Train digit net
            #---------------------

            opt_digit.zero_grad()

            pred_digit, _ = audio_digit_net(x)
            audio_digit_loss = digit_loss(pred_digit, digit)
            audio_digit_loss.backward()
            opt_digit.step()

            #---------------------------------------
            # Calculate accuracies on training set
            #---------------------------------------

            predicted = torch.argmax(pred_gender.data, 1)
            correct_gender += (predicted == gender).sum()

            predicted = torch.argmax(pred_digit.data, 1)
            correct_digit += (predicted == digit).sum()

        train_accuracy_gender = 100 * correct_gender / n_train
        train_accuracy_digit = 100 * correct_digit / n_train
        writer.add_scalar("train_digit_acc", train_accuracy_digit, epoch + 1)
        writer.add_scalar("train_gender_acc", train_accuracy_gender, epoch + 1)

        #---------------------------------------
        # Evaluate model on test set
        #---------------------------------------

        correct_gender = 0
        correct_digit = 0
        accum_loss_digit = 0
        accum_loss_gender = 0

        for i, (x, gender, digit, _) in enumerate(test_loader):
            audio_digit_net.eval()
            audio_gender_net.eval()
            x = torch.unsqueeze(x,1).to(device)
            digit = digit.to(device)
            gender = gender.to(device)

            pred_digit, _ = audio_digit_net(x)
            pred_gender, _ = audio_gender_net(x)

            audio_gender_loss_val = gender_loss(pred_gender, gender)
            audio_digit_loss_val = digit_loss(pred_digit,digit)

            accum_loss_digit+=audio_digit_loss
            accum_loss_gender+=audio_gender_loss

            predicted = torch.argmax(pred_gender.data, 1)
            correct_gender += (predicted == gender).sum()

            predicted = torch.argmax(pred_digit.data, 1)
            correct_digit += (predicted == digit).sum()

        test_accuracy_gender = 100 * correct_gender / n_test
        test_accuracy_digit = 100 * correct_digit / n_test
        writer.add_scalar("test_digit_acc", test_accuracy_digit, epoch + 1)
        writer.add_scalar("test_gender_acc", test_accuracy_gender, epoch + 1)

        print("Epoch {} completed | Time: {:5.2f} s".format(epoch + 1, time.time() - epoch_start))
        print("Digit | Train set accuracy: {} % | Test set accuracy: {} %".format(train_accuracy_digit, test_accuracy_digit))
        print("Gender | Train set accuracy: {} % | Test set accuracy: {} %".format(train_accuracy_gender, test_accuracy_gender))
        print("#____________________________________________________________#")

        if lowest_loss_gender > accum_loss_gender:
            best_test_acc_gender = test_accuracy_gender
            torch.save(audio_gender_net.state_dict(),os.path.join(root, 'audio_gender_net_early_stop_epoch_{}.pt'.format(epoch)))
            lowest_loss_gender = accum_loss_gender
            counter_gender=0
        else:
            counter_gender +=1


        if lowest_loss_digit > accum_loss_digit :
            best_test_acc_digit = test_accuracy_digit
            torch.save(audio_digit_net.state_dict(),os.path.join(root, 'audio_digit_net_early_stop_epoch_{}.pt'.format(epoch)))
            lowest_loss_digit = accum_loss_digit
            counter_digit=0
        else:
            counter_digit+=1

        if counter_gender > args.stop_interval:
            lowest_loss_gender = -1
            final_acc_gender = test_accuracy_gender
            print(final_acc_gender)
            print('Not training gender more')

        if counter_digit > args.stop_interval:
            lowest_loss_digit = -1
            final_acc_digit = test_accuracy_digit
            print(final_acc_digit)
            print('Not training digit more')

        if lowest_loss_digit ==-1 and lowest_loss_gender==-1:
            exit()

if __name__ =="__main__":
    main()
