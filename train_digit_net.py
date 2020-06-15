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

LongTensor = torch.cuda.LongTensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default = None)
    parser.add_argument("--epochs", type = int, default = 2000)
    parser.add_argument("--model", type = str, default = None)
    parser.add_argument("--device", type = str, required = True)
    parser.add_argument("--experiment_name", type = str, required = True)
    parser.add_argument("--stop_interval", type = int, default = 100)


    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    root = Path(args.root)
    experiment_name = args.experiment_name
    architecture = args.model
    device = 'cuda:' + args.device

    log_dir = os.path.join(root,'logs')
    exp_dir = os.path.join(log_dir,experiment_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(exp_dir))


    # Meta data and list of data files

    annotation_file = '/home/adam/adversarial_learning_speech/audio_mnist/audio_mnist/audioMNIST_meta.json'
    train_file_index = librosa.util.find_files('/home/adam/adversarial_learning_speech/audio_mnist/audio_mnist/')

    split_ratio = 5

    # Build indices for the data
    file_index, annotation_index_gender, annotation_index_digit, annotation_index_speaker_id = dataset.build_annotation_index(
                                                                    train_file_index, annotation_file, balanced_genders = False)
    val_annotation_index, train_annotation_index, val_ids, train_ids = dataset.balanced_annotation_split(file_index, annotation_index_gender, annotation_index_digit, annotation_index_speaker_id, split_ratio)

    # Some hyper parameters
    sampling_rate = 8000
    segment_length = 8192
    num_classes = 10
    num_epochs = args.epochs
    batch_size = 100

    lamb = 10000
    eps = 1e-4

    # Create the dataset
    train_data = dataset.AnnotatedAudioDataset(train_annotation_index, sampling_rate, segment_length
    )

    val_data = dataset.AnnotatedAudioDataset(
        val_annotation_index, sampling_rate, segment_length
    )

    n_train = train_data.__len__()
    n_val = val_data.__len__()

    #Set up models
    fft = Audio2Mel(sampling_rate = 8000)
    if architecture == 'resnet':
        discriminator_digit = load_modified_ResNet(num_classes)
        discriminator_digit.to(device)
    elif architecture == 'alexnet':
        discriminator_digit = load_modified_AlexNet(num_classes)
        discriminator_digit.to(device)


    # Loss and optimizer
    adversarial_loss = nn.CrossEntropyLoss()
    optimizer_disc = torch.optim.Adam(discriminator_digit.parameters(), lr = 1e-4, betas = (0.5, 0.9))



    # Dataloaders
    train_loader = DataLoader(train_data, batch_size = 64, num_workers = 2, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = 10, num_workers = 2, shuffle = True)

    lowest_loss = 1e+6
    iter = 0
    best_val_acc = 0
    counter_digit=0

    print("Training initiated.")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        correct = 0
        for i, (x, gender, digit, _) in enumerate(train_loader):
            discriminator_digit.train()
            optimizer_disc.zero_grad()
            x = torch.unsqueeze(x,1)
            digit = digit.to(device)

            # Audio to spectrogram
            spectrograms = fft(x).detach()

            spectrograms, means, stds = preprocess_spectrograms(spectrograms)
            spectrograms = torch.unsqueeze(spectrograms,1).to(device)

            out = discriminator_digit(spectrograms)
            disc_loss = adversarial_loss(out,digit)

            disc_loss.backward()
            optimizer_disc.step()

            predicted = torch.argmax(out.data, 1)
            correct += (predicted == digit).sum()

            iter += 1
            writer.add_scalar("d_loss", disc_loss.item(),iter)

        train_accuracy  = 100*correct / n_train
        writer.add_scalar("Train_set_accuracy", train_accuracy, epoch+1 )


        loss_accum = 0
        correct = 0
        discriminator_digit.eval()
        for i, (x, gender, digit, _) in enumerate(val_loader):
            x = torch.unsqueeze(x,1)
            digit = digit.to(device)
            spectrograms = fft(x).detach()
            spectrograms, means, stds = preprocess_spectrograms(spectrograms)
            spectrograms = torch.unsqueeze(spectrograms,1).to(device)
            out = discriminator_digit(spectrograms)
            disc_loss = adversarial_loss(out,digit)

            predicted = torch.argmax(out.data, 1)
            correct += (predicted == digit).sum()

            loss_accum+=disc_loss.item()
        val_accuracy = 100 * correct / n_val
        writer.add_scalar("val_set_accuracy", val_accuracy, epoch+1)
        writer.add_scalar("val_set_loss", loss_accum,epoch+1)
        print("Epoch {} completed | Time: {:5.2f} s | Train set accuracy: {} % | val set accuracy: {} %".format(
                epoch + 1,
                time.time() - epoch_start,
                train_accuracy,
                val_accuracy
        ))


        if lowest_loss > loss_accum:
            best_val_acc = val_accuracy
            torch.save(discriminator_digit.state_dict(),os.path.join(root, 'best_digit_alexnet_spectrograms_epoch_{}.pt'.format(epoch)))
            lowest_loss = loss_accum
            counter_digit=0
        else:
            counter_digit+=1

        if counter_digit > args.stop_interval:
            print('Best accuracy:{}'.format(best_val_acc))
            exit()


if __name__ == "__main__":
    main()
