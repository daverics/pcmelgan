# Instructions to train the Private Conditional MelGAN
In this README we include an overview of the file structure, a description on where to download the AudioMnist data used in this study, how to setup the environment to run the scripts in, and how to run the experiments and produce results.

If you find this work useful, please cite us.
```bibtex
@article{ericsson2020adversarial,
  title={Adversarial representation learning for private speech generation},
  author={Ericsson, David and {\"O}stberg, Adam and Zec, Edvin Listo and Martinsson, John and Mogren, Olof},
  journal={2020 Workshop on Self-supervision in Audio and Speech at the 37th International Conference on Machine Learning},
  year={2020}
}
```


## File structure



	├── fixed_classifiers
	│   ├── audio_digit_net_early_stop_epoch_26.pt # Digit audio
 	│   ├── audio_gender_net_early_stop_epoch_36.pt # Gender audio
 	│   ├── best_digit_alexnet_spectrograms_epoch_26.pt # Digit spectrogram
	│   ├── best_gender_alexnet_epoch_29.pt # Gender classifier spectrogram
	├── logs # save dir for experiments
	├── mel2wav
	│   ├── best_netG_epoch_2120.pt # MelGAN checkpoint
	│   ├── interface.py # Defines MelGAN model
	│   ├── modules.py # Defines the audio to mel spectrogram
	│   ├── utils.py # Save samples and preprocess spectrograms
	├── dataset.py # Dataset for audio files
	├── evaluate_experiment_pcgan.py
	├── filter.py # Defines the U-Net model
	├── generate_plots.py
	├── models.py # Defines AudioNet, modified AlexNet and modified Resnet
	├── run_experiment_pcgan.py
	├── train_audio_net.py # Trains AudioNet digit and gender classifiers
	├── train_digit_net.py # Trains AlexNet/ResNet spec. digit classifier
	├── train_gender_net.py # Train AlexNet/ResNet spec. gender classifier
	├── utils.py



## Download data
The AudioMnist dataset can be found at: https://github.com/soerenab/AudioMNIST


## Setup environment (anaconda)
Install anaconda.

	conda env create -f environment.yml
	conda activate pcmelgan


## Train the models

	python run_experiment_pcgan.py --experiment_name example_name --device 0


## Evaluate the models
To evaluate the main method experiment:

  python evaluate_experiment_pcgan.py --experiment_name example_name



## Generate plots for the evaluation
To generate plots for the run:

	python generate_plots.py --experiment_paths example_path --result_dir example_path --model example_folder_name


## Audio samples
We provide samples from the AudioMNIST test set that were transformed by our model. The shared folder contains original sound clips and their corresponding transformed versions, with both male and female versions. Consider the sample from speaker 36 saying digit 1 as an example, the transformed samples as well as the original sample are as follow:
PCMelGAN with sampled gender male: speaker_36_digit_1_sampled_gender_male.wav  
PCMelGAN with sampled gender female: speaker_36_digit_1_sampled_gender_female.wav  
Filter: speaker_36_digit_1_filtered.wav  
Original: speaker_36_digit_1_original.wav  

Link to samples: https://www.dropbox.com/sh/oangx84ibhzodhs/AAAfG-PBW4Ne8KwdipAmKFy1a?dl=0
