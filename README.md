# SNN_EEG  
This repository contains sample code for training SNN towards producing artificial Motor Imagery EEG signals.  
The process is implemented in three stages:  
 1. Template exraction  
 2. Training SNN with template  
 3. Aritificial data generation  
  
A sample MI dataset, which was processed, is saved to data folder.  
  
To complete the process, run the following scripts in order.  
## Template extraction  

    cd data 
    python extract_template.py 
   This scripts saves the best trial of the dataset as *.npy* in *data* folder.
 
## Training SNN  
  
    python  snn_train.py
This create two folder, "saveddata", "syntheticdata" in main directory.
Also the input is saved in the main directory, with an extention *_input_spike_data
*NOTE:* Do not delete the directoried or the saved input data if data generation is needed

## Data generation
run the following command

    python generate_synthetic_data.py
Run this command only after training the SNN (previous).

### Note
The pseduoderivative and nomenclature of the spiking variable were sourced from [Zenke github](https://github.com/fzenke/spytorch)

