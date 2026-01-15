import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # shrink TensorFlow logger verbosity
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")  # ask TensorFlow for deterministic GPU kernels when possible

import pdb
from re import L
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.model_selection import train_test_split

from data.data_processing import length_scanner, load_data, one_hot_encoder, \
    Data_Generator, int_to_smiles, seq_cleaning, smiles_to_int, smiles_cleaning, \
    amino_to_int, protein_one_hot_encoder
from models.generative_models import lstm_model, generate, generate_batch
from utils.utils import apply_thresholds, selfies_one_hot_encoder, \
    smiles2selfies, write_report, plot_acc_loss, compute_properties, histogram_properties, tanimoto2, \
    compute_tanimoto_init, filter_unique_molecules, apply_structural_filters, apply_unwanted_substruct_filters, \
    apply_trifurcated_filter

import random
import numpy as np
import argparse as ap
import pandas as pd
import glob

from tqdm import tqdm

import time
import rdkit.Chem as Chem
from rdkit import RDLogger
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

# Hide routine RDKit logs in stderr (parsing and kekulization noise from invalid SMILES)
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    # Fallback if the TensorFlow build does not expose deterministic kernels
    pass


def configure_seeds(seed_value: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def parseArg():
    '''
    Parse all the parameters
    '''

    parser = ap.ArgumentParser()

    parser.add_argument("-g", "--general_set_smiles", required=True, type=str, help="path to general set of smiles file.")
    parser.add_argument("-e", "--epochs", required=True, type=int, default=100, help="number of epochs (default: 100).")
    parser.add_argument("-v", "--validation_size", required=True, type=float, default=0.1, help="validation set size (default: 0.1).")
    parser.add_argument("-t", "--test_size", required=True, type=float, default=0.1, help="test set size (default: 0.1).")
    parser.add_argument("-b", "--batch_size", required=True, type=int, default=100, help="batch size for general training (default: 100).")
    parser.add_argument("-s", "--sampling_temperature", required=False, type=float, default=1.2, help="sampling temperature (default: 1.2).")
    parser.add_argument("-q", "--quantity", required=True, type=int, help="target amount defined by --quantity-mode (valid molecules or total attempts).")
    parser.add_argument("-n", "--name", required=True, type=str, help="name of generative run.")
    parser.add_argument("-o", "--outdir", required=True, type=str, help="output directory.")
    parser.add_argument("-pa", "--specific_set_smiles", required=False, type=str, help="path to alternative smiles file.")
    parser.add_argument("-pt", "--pretrained", required=False, type=str, help="restart training from stored weights. Path to weights.")
    parser.add_argument("-ial", "--inner_al", required=False, type=int, help="number of 'al' active learning iterations.")
    parser.add_argument("-qed", "--druglikeness", required=False, type=float, help="threshold for qed value of generated molecules.")
    parser.add_argument("-sa", "--sascore", required=False, type=float, help="threshold for sascore value of generated molecules.")
    parser.add_argument("-ta", "--tanimoto", required=False, type=float, help="threshold for tanimoto score of generated molecules.")
    parser.add_argument("-tad", "--ta_direction", required=False, type=str, default='<', choices=['<', '>'], help="direction of the tanimoto threshold filter (default: '<').")
    parser.add_argument("-tp", "--top_p", required=False, type=float, help="nucleus (top-p) sampling cutoff (0-1)")
    parser.add_argument("--disable_top_p", action="store_true", help="disable nucleus sampling and rely solely on temperature sampling.")
    parser.add_argument("-sub", "--subset_optimizer", required=False, type=str, help="if optimizer, try the subset where to compute tanimoto similarity.")
    parser.add_argument("-mw", "--molecular_weight", required=False, nargs=2, type=float, metavar=('MIN', 'MAX'), help="filter of molecular weight range, e.g. --mw 20 500")
    parser.add_argument("-r", "--restart", required=False, action="store_true", help="restart job from last stop.")
    parser.add_argument("-trifurcated_filter", "--trifurcated_filter", required=False, type=int, help="Apply trifurcated substructure filter, indicating the min branch size if it is applied (False or int).")
    parser.add_argument("-acf", "--apply_catalog_filter", action="store_true", help="Run the PAINS/Brenk/NIH/CHEMBL substructure filter step (skipped by default).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible training/generation. Omit to allow stochastic runs.")
    parser.add_argument("--scale", type=float, default=0.5, help="Latent space noise scale factor for generation (default: 0.5). Lower = more valid/less diverse; higher = more diverse/more invalid.")
    
    args = parser.parse_args()

    if args.seed is not None:
        configure_seeds(args.seed)
    
    # If inner active learning, thresholds for druglikeness, sascore, and tanimoto are required
    if args.inner_al and not all([args.druglikeness, args.sascore, args.tanimoto]):
        parser.error("--inner_al requires --druglikeness, --sascore, and --tanimoto.")
        
    # Choose or restart or pretrained, but not both
    if args.restart and args.pretrained:
        parser.error("--restart and --pretrained can't run simultaneously.")
        
    # If tanimoto, choose direction of the filter
    if args.tanimoto and not args.ta_direction:
        parser.error("--tanimoto requires --ta_direction to indicate the direction of the tanimoto filter.")
        
    # If ta direction is ">", provide the initial specific set, because it compares the tanimoto against it
    if args.ta_direction == '>' and args.restart and not args.specific_set_smiles:
        parser.error("--ta_direction > and --restart requires --specific_set_smiles to compute the tanimoto against it")

    # If there is a subset for optimizer generation, the ta direction should always be >
    if args.subset_optimizer and not args.ta_direction == '>':
        parser.error("--subset_optimizer only works when --ta_direction is > because we aim to optimize\
                    the generation by computing the tanimoto similarity against this personalized subset")
        
    # Define variables
    general_smiles = args.general_set_smiles
    num_epochs = args.epochs
    val_size = args.validation_size
    test_size = args.test_size
    batch_size = args.batch_size
    sampling_temperature = args.sampling_temperature
    quantity = args.quantity
    name = args.name
    outdir = args.outdir
    specific_smiles = args.specific_set_smiles
    weights = args.pretrained
    inner_al = args.inner_al
    qed_t = args.druglikeness
    sascore_t = args.sascore
    tanimoto_t = args.tanimoto
    ta_direction = args.ta_direction
    subset = args.subset_optimizer
    mw = args.molecular_weight
    restart = args.restart
    skip_third_filter = not args.apply_catalog_filter
    top_p = None if args.disable_top_p else args.top_p
    if top_p is not None and not 0 < top_p <= 1:
        parser.error("--top_p must be in the interval (0, 1].")
    
    trifurcated = args.trifurcated_filter

    return general_smiles, num_epochs, val_size, test_size, batch_size, sampling_temperature, top_p, quantity, \
    name, outdir, specific_smiles, weights, inner_al, qed_t, sascore_t, tanimoto_t, ta_direction, subset, \
    mw, skip_third_filter, trifurcated, restart, args.seed, args.scale


def split_data(dat, validation_size, test_size, maximum_len, char_alph=None, random_state=None):
    '''
    Args:
    - dat: The input data to be split
    - validation_size: The proportion of the data to be used for validation
    - test_size: The proportion of the data to be used for testing
    - maximum_len: Max length
    - char_alph: List of the unique characters found in a SMILE.
    - random_state: Random seed for reproducible splits. If None, splits will be random.

    Operations:
    - Split the dataset into training, validation and test sets.
    - One hot encoding of the predictors (x) and target (y)

    Return:
    - All sets
    '''

    # Split the dataset into train, test and validation
    pre_train, validation = train_test_split(dat, test_size=validation_size, random_state=random_state)
    train, test = train_test_split(pre_train, test_size=test_size, random_state=random_state)
    
    print('train: ' , type(train))
    print("\nTraining set size: ", train.shape[0])
    print("Validation set size: ", validation.shape[0])
    print("Test set size: ", test.shape[0])
    print("Number of possible characters: ", len(int_to_smiles))

    # One-hot-encoding. The resulting object is a 3D Tensor (Number of SMILES x Length of SMILES x Number of characters in a SMILE) len(smiles_to_int)
    train_pred, train_target = one_hot_encoder(smiles=train, max_len=maximum_len, num_char=len(smiles_to_int))
    val_pred, val_target = one_hot_encoder(smiles=validation,  max_len=maximum_len, num_char=len(smiles_to_int))
    test_pred, test_target = one_hot_encoder(smiles=test,  max_len=maximum_len, num_char=len(smiles_to_int))

    return train, validation, test, train_pred, train_target, val_pred, val_target, test_pred, test_target


def train_and_evaluate_model(model, train_pred, train_target, val_pred, val_target, test_pred, test_target, b_size, n_epochs, output_directory, name, checkpoint_name, moment, deb):
    '''
    Args:
    - model: The model to be trained and evaluated.
    - train_pred: The input data for training.
    - train_target: The ground truth labels for training.
    - val_pred: The input data for validation.
    - val_target: The ground truth labels for validation.
    - test_pred: The input data for testing.
    - test_target: The ground truth labels for testing.
    - b_size: The batch size used during training.
    - n_epochs: The number of epochs for training.
    - output_directory: The directory to save checkpoints and plots.
    - name: The name prefix for saved files.
    - checkpoint_name: The name of the checkpoint file.
    - moment: The moment of the training (Start, Transfer, Restart)
    - deb: Debugging mode (run_eagerly)

    Operations:
    - Train the model using the given generators, evaluate its performance and save relevant information (checkpoint)

    Returns:
    - score: The evaluation loss on the test set.
    - acc: The evaluation accuracy on the test set.
    '''

    # Number of steps in each epoch
    steps_per_epoch = len(train_pred) // b_size

    # Generate a learning rate schedule
    # lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    #     initial_learning_rate=1e-3,           
    #     decay_steps=steps_per_epoch * 50,
    #     decay_rate=0.5,
    #     staircase=True
    # )
    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=steps_per_epoch * 100,
        t_mul=1.0,
        m_mul=1.0,
        alpha=0.0
    )
    opt = Adam(learning_rate=lr_schedule, clipnorm=1.0)  # clip gradients to norm=1

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['acc'],
        run_eagerly=False
    )

    training_generator = Data_Generator(train_pred, train_target, b_size)
    validation_generator = Data_Generator(val_pred, val_target, b_size)

    # Define checkpoint
    if moment == 'Start':
        checkpoint = ModelCheckpoint(output_directory + '/' + name + checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', save_freq='epoch')
        callbacks_list = [checkpoint]

    elif moment == 'Transfer':
        print('Model checkpoint: ', output_directory + '/' + name + checkpoint_name)
        checkpoint = ModelCheckpoint(output_directory + '/' + name + checkpoint_name, monitor='val_loss', verbose = 1, save_weights_only = True, mode ='auto', save_freq='epoch')
        callback =  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
        callbacks_list = [callback]
    
    elif moment == 'Restart':
        checkpoint = ModelCheckpoint(output_directory + '/' + name + checkpoint_name, monitor='val_loss', verbose = 1, save_weights_only = True, mode ='auto')
        callbacks_list = [checkpoint]

    # Train the model
    validation_steps = len(val_pred) // b_size

    # add ReduceLROnPlateau
    plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-6
    )
    callbacks_list.append(plateau)

    # you still keep your ModelCheckpoint / EarlyStopping …
    history = model.fit(
        training_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=len(val_pred)//b_size,
        shuffle=True,
        callbacks=callbacks_list
    )

    # Generate the accuracy and loss plots
    plot_acc_loss(history=history, name=name, output_dir=output_directory)

    # Evaluate the model
    score, acc = model.evaluate([test_pred, test_pred], test_target, batch_size=b_size, verbose=0)        

    return score, acc


def computing_properties_spec_set(dat, output_dir, name):
    '''
    Args:
    - dat: Specific set with the compounds
    - output_dir: Name of the output directory
    - name: Base name of the CSV file generated

    Operations:
    - Compute the properties of the compounds of the specific set
    - Save them in a dataframe

    Returns:
    - Dataframe containing the properties of each compound of the specific set
    '''

    data_init = []
    print('Computing properties for specific set...')
    
    # Compute the properties of each compound
    for comp in tqdm(dat.tolist()):
        try:
            smi, sa, qed, mw, logp, tpsa, nhd, nha = compute_properties(comp)
            data_init.append([smi, sa, qed, mw, logp, tpsa, nhd, nha])
        except:
            pass # if not valid compound
        
    # Save the properties in a dataframe
    df_init_props = pd.DataFrame(data_init, columns=['smiles', 'SAscore','QED', 'mw', 'logp', 'tpsa', 'nHD', 'nHA'])
    df_init_props.to_csv(output_dir + '/' + name + '_specific_smiles.csv')
    
    return df_init_props


def computing_properties_generated_set(smiles, t_smiles, output_dir, name, df_init_props, mom, ta_direction):
    '''
    Args:
    - smiles: Generated compounds + specific training set 
    - t_smiles:
    - output_dir: Name of the output directory
    - name: Base name of the CSV file generated
    - df_init_props: Properties of the specific set
    - mom: The moment of the training (Start, Transfer, Restart)
    - ta_direction: The direction of the tanimoto threshold filter.

    Operations:
    - Compute properties of the compounds of the generated compounds
    - Generate histograms for the comparison of properties (specific vs generated)
    - Save them in a dataframe

    Return:
    - Dataframe containing the properties of each compounds of the general set
    '''

    data = []
    print('Computing properties...')

    if mom == 'restart':
        smiles_fps = smiles

    elif mom == 'initial':
        smiles_fps = compute_tanimoto_init(smiles)

    # Compute properties of the general set 
    for comp in tqdm(t_smiles):
        try:
            smi, sa, qed, mw, logp, tpsa, nhd, nha = compute_properties(comp)

            if mom == 'initial' or 'restart':
                if ta_direction == '<':
                    tan_mean, tan_max = tanimoto2(comp, smiles_fps, ta_direction)
                    sims = []
                    data.append([smi, sa, qed, mw, logp, tpsa, nhd, nha, tan_mean, tan_max, sims])
                elif ta_direction == '>':
                    tan_mean, tan_max, sims = tanimoto2(comp, smiles_fps, ta_direction)
                    data.append([smi, sa, qed, mw, logp, tpsa, nhd, nha, tan_mean, tan_max, sims])
                else:
                    raise ValueError('Indicate the ta_direction: < or >')

            else:
                data.append([smi, sa, qed, mw, logp, tpsa, nhd, nha])

        except:
            pass
    
    # Save the properties in a dataframe
    if mom == 'initial' or 'restart':
        df_gen_props = pd.DataFrame(data, columns=['smiles', 'SAscore','QED', 'mw', 'logp', 'tpsa', 'nHD', 'nHA', 'tan_mean', 'tan_max', 'sims'])
        
        # Generate histograms
        for prop in list(df_gen_props.columns)[1:-3]:
            histogram_properties(df_init_props, df_gen_props, prop, output_dir)

    else:
        df_gen_props = pd.DataFrame(data, columns=['smiles', 'SAscore','QED', 'mw', 'logp', 'tpsa', 'nHD', 'nHA'])


    # Save the dataframe
    df_gen_props_simple = df_gen_props[['smiles', 'SAscore','QED', 'mw', 'logp', 'tpsa', 'nHD', 'nHA', 'tan_mean', 'tan_max']]
    df_gen_props_simple.to_csv(output_dir + '/' + name + '_generated_smiles.csv')
    print("\nSmiles written to " + output_dir + '/' + name + '_generated_smiles.csv')

    return df_gen_props

def model_vae(model, size, test_pred, scale=0.5):
    '''
    Args:
        model (tf.keras.Model): The VAE model from which components will be extracted.
        size (int): The size of the vocabulary.
        test_pred (numpy.ndarray): The test data used to predict the latent space representations.
        scale (float): The scale factor for latent space noise (default: 0.5).

    Operations:
        1. Extract the encoder part of the VAE model.
        2. Create a model to map latent space vectors to LSTM states.
        3. Create a decoder model for sequence generation.
        4. Copy weights from the original VAE model to the decoder model.
        5. Predict the latent space representation for the test data.
        6. Randomly select a latent space vector as the seed for sequence generation.
        7. Use the provided scale factor for sequence generation.

    Returns:
        gen_model (tf.keras.Model): The decoder model for sequence generation.
        latent_to_states_model (tf.keras.Model): The model mapping latent space vectors to LSTM states.
        latent_seed (numpy.ndarray): The seed for generating sequences.
        scale (float): The scale factor for sequence generation.
    '''

    encoder = Model(inputs=model.layers[0].input, outputs=model.layers[3].output)
    
    latent_input = Input(shape=(128, ))
    state_h = model.layers[5](latent_input)
    state_c = model.layers[6](latent_input)
    latent_to_states_model = Model(latent_input, [state_h, state_c])
    print('\n')
    latent_to_states_model.summary()

    decoder_inputs = Input(batch_shape=(1, 1, size))
    decoder_lstm = LSTM(256, return_sequences=True, stateful=True)(decoder_inputs)
    decoder_outputs = Dense(size, activation='softmax')(decoder_lstm)
    gen_model = Model(decoder_inputs, decoder_outputs)

    for i in range(1,3):
        gen_model.layers[i].set_weights(model.layers[i+6].get_weights())

    print('\n')
    gen_model.summary()

    test_latent_space = encoder.predict(test_pred)

    rand_seed = random.randrange(test_latent_space.shape[0])
    latent_seed = test_latent_space[rand_seed:rand_seed + 1]

    return gen_model, latent_to_states_model, latent_seed, scale

def main():
    
    time0 = time.time()
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print("Error setting GPU config:", e)
    
    # Set all variables to the given argument values
    general_smiles, num_epochs, val_size, test_size, batch_size, sampling_temp, top_p, quant, name, outdir,\
    specific_smiles, weights_file, inner_al, qed_t, sascore_t, tanimoto_t, ta_direction, subset, mw, skip_third_filter,\
    trifurcated, restart, seed_value, scale_value = parseArg()

    
    # Print relevant information to log
    print('PARAMETERS:')
    print('  - General training set:', general_smiles)
    print('  - Specific training set:', specific_smiles)
    print('  - Weights file from pretrained:', weights_file)
    print('  - Name:', name)
    print('  - Output directory:', outdir)
    print('  - Num epochs:', num_epochs)
    print('  - Validation size:', val_size)
    print('  - Test size:', test_size)
    print('  - Batch size:', batch_size)
    print('  - Sampling Temperature:', sampling_temp)
    print('  - Top-p (nucleus) sampling:', 'disabled' if top_p is None else top_p)
    print('  - Latent noise scale:', scale_value)
    print('  - Quantity:', quant)
    print('  - Seed:', seed_value)
    print('  - Inner Active Learning Cycles:', inner_al)
    print('  - QED threshold:', qed_t)
    print('  - SAscore threshold:', sascore_t)
    print('  - Tanimoto threshold:', tanimoto_t)
    print('  - Tanimoto direction:', ta_direction)
    print('  - Subset optimizer:', subset)
    print('  - Molecular weight:', mw)      
    print('  - Skip third filter (PAINS/Brenk/NIH/CHEMBL):', skip_third_filter)
    print('  - Trifurcated filter:', 'disabled' if not trifurcated else f'enabled (min branch size: {trifurcated})')
    print('\n')
    
    # Generate the output directory
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    inner_output_dir = outdir + '/' + name

    if restart:
        print('Restarting the job')
        # Define paths according to the rounds that have been done previously
        round = glob.glob('%s_*'%inner_output_dir)
        round = [os.path.isdir(x) for x in round]
        round = len(round)
        print(f'\nRounds done: {round-1}\n')
    else:
        round = 1
        if not os.path.exists(inner_output_dir + '_' + str(round)):
            os.mkdir(inner_output_dir + '_' + str(round))

    # Generate a plot with the length of the SMILES of the general training set and return 2*median of the length of the SMILES
    max_len = length_scanner(general_smiles)

    # Transform each of the SMILES from the general training set into an array. The ones that have a length higher than max_len will be removed
    data_general = load_data(data=general_smiles, max_len=max_len)

    # Clean SMILES from salts and charges and standardize them
    data_general = smiles_cleaning(data=data_general)

    # Number of characters in SMILES (==50)
    n_vocab = len(smiles_to_int)
    max_generation_len = int(max_len)

    # Split the general training dataset into train, test and validation one-hot-encoded
    # Use fixed split seed (42) for fair comparisons across different training seeds
    train, validation, test, train_x, train_y, val_x, val_y, test_x, test_y = split_data(data_general, val_size, test_size, max_len, random_state=42)
    
    # Store original data sizes for reporting
    original_train_size = train.shape[0]
    original_val_size = validation.shape[0]
    original_test_size = test.shape[0]
    
    # Define the LSTM VAE model architecture based on the input and output dimensions of the data
    model = lstm_model(train_x, train_y)
    model.summary()
    
    # If pretrained weights file is provided, load these weights. If not, run the
    # training of the general training set and then load the resulting weights.
    if not weights_file and not restart:
        print('Training of the general set...')        
        training_dir = outdir + '/pretrained'
        if not os.path.exists(training_dir):
            os.mkdir(training_dir)
            
        train_and_evaluate_model(model, train_x, train_y, val_x, val_y, test_x, test_y, batch_size, num_epochs, training_dir, 'general', '_weights.hdf5', 'Start', True)
        weights_file = training_dir + '/general_weights.hdf5'
    
    initial_specific = specific_smiles
    
    timegt = time.time() - time0
    print('Time training general: ', timegt, 'seconds')
    print('Time human readable:', time.strftime("%H:%M:%S", time.gmtime(timegt)))

    # For each inner cycle of active learning:
    for i in range(round, inner_al + 1):
        timeal0 = time.time()

        # TODO: What happens if no specific set (specific_smiles) is provided? It should continue generating but not targeted. Implement!!
        
        if restart:
            # define specific set from the previous round
            specific_smiles = inner_output_dir + '_' + str(i) + '/' + name + '_generated_smiles_merged.csv'
            print('Specific set: ', specific_smiles)
                
        # Transform each of the SMILES from the specific training set into an array. The ones that have a length higher than max_len will be removed
        data_specific = load_data(data=specific_smiles, max_len=max_len)
        print('Load_data: ', data_specific)
        data_specific = smiles_cleaning(data=data_specific)
        print('Smiles_cleaning: ', len(data_specific))
        
        # Compute properties of the specific training set
        df_init_props = computing_properties_spec_set(data_specific, inner_output_dir+'_'+str(i), name)
        
        # Split the specific training set into train, test and validation one-hot-encoded
        # Use fixed split seed (42) for fair comparisons across different training seeds
        train, validation, test, train_x, train_y, val_x, val_y, test_x, test_y = split_data(data_specific, val_size, test_size, max_len, random_state=42)
        n_vocab = len(smiles_to_int)
        
        print(f'\nROUND {i}')
        
        # Load weights
        if round == 1:
            print('loading weights for first round ...')
            model.load_weights(weights_file)
        else:
            print('loading weights for round %s...'%round)
            model.load_weights(inner_output_dir + '_1' + '/' + name + '_weights.hdf5')
        
        # Batch size is all the specific training set
        batch_size = len(val_x) - 2    # Specific batch size not given by parser!
        num_epochs = 100               # Specific num epochs not given by parser!
        
        # Train, evaluate the model and generate checkpoints using the specific training set
        score, acc = train_and_evaluate_model(model, train_x, train_y, val_x, val_y, test_x, test_y, batch_size, num_epochs, inner_output_dir+'_'+str(i), name, '_weights.hdf5', 'Transfer', True)
        
        print(f'\nTest alternative loss: {score}')
        print(f'Test alternative accuracy: {acc}')
        
        model.save_weights(inner_output_dir+'_'+str(i) + '/' + name + '_weights.hdf5')
        
        gen_model, latent_to_states_model, latent_seed, scale = model_vae(model, len(int_to_smiles), test_x, scale=scale_value)

        # Evaluate the model
        score, acc = model.evaluate([test_x, test_x], test_y, batch_size=batch_size, verbose=0)
        print(f'\nTest loss: {score}')
        print(f'Test accuracy: {acc}')
        
        timest = time.time() - timeal0
        print('Time training specific: ', timest, 'seconds')
        print('Time human readable:', time.strftime("%H:%M:%S", time.gmtime(timest)))
        
        # Generate new compounds with nucleus (top-p) sampling until N valids
        timeg0 = time.time()
        attempt_records = []
        t_mols, t_smiles, not_valid, total_attempts = generate_batch(
            gen_model, latent_to_states_model, latent_seed,
            sampling_temp=sampling_temp, n_vocab=n_vocab,
            scale=scale, quant=quant,
            batch_size=batch_size, top_p=top_p,
            attempt_recorder=attempt_records,
            max_len=max_generation_len
        )

        attempts_path = inner_output_dir + '_' + str(i) + '/' + name + f'_generated_smiles_all_attempts_round_{i}.csv'
        pd.DataFrame(attempt_records).to_csv(attempts_path, index=False)
        print(f"All generation attempts written to {attempts_path}")

        print('GENERATED COMPOUNDS')
        print(len(t_smiles))
        
        timeg = time.time() - timeg0
        print('Time generation: ', timeg)
        print('Time human readable:', time.strftime("%H:%M:%S", time.gmtime(timeg)))

        if len(t_mols)<=0:
            raise Exception("No valid compounds generated.")
        
        valid_generated = len(t_mols)
        print(f"Collected {valid_generated} valid molecules (attempts={total_attempts}).")

        write_report(train_size=original_train_size, val_size=original_val_size, test_size=original_test_size, test_loss=score,
            test_acc=acc, epochs=num_epochs, batch_size=batch_size, sampling_temp=sampling_temp, top_p=top_p, quantity_requested=quant,
            valid_generated=valid_generated, total_attempts=total_attempts,
            not_valid=not_valid, name=name, outdir=inner_output_dir+'_'+str(i), seed=seed_value)
        
        timef0 = time.time()

        # Apply 1st filter: Substructural filters
        df_t_smiles = pd.DataFrame({'smiles': t_smiles, 'mols': t_mols})
        df_t_smiles = apply_structural_filters(df_t_smiles)
        t_smiles = df_t_smiles['smiles'].tolist()
        print('FIRST FILTER:')
        print(len(t_smiles))
        
        if len(t_smiles)==0:
            raise Exception("No valid compounds generated after the first filter.")
        
        # Compute properties of the generated compounds
        if ta_direction == '<':
            df_gen_props = computing_properties_generated_set(specific_smiles, t_smiles, inner_output_dir+'_'+str(i), name, df_init_props, 'initial', ta_direction)
        elif ta_direction == '>':
            if subset:
                df_gen_props = computing_properties_generated_set(subset, t_smiles, inner_output_dir+'_'+str(i), name, df_init_props, 'initial', ta_direction)
            else:
                df_gen_props = computing_properties_generated_set(initial_specific, t_smiles, inner_output_dir+'_'+str(i), name, df_init_props, 'initial', ta_direction)
        else:
            raise ValueError('Indicate the ta_direction: < or >')
    
        # Apply 2nd filter: Thresholds (QED, SA and Tanimoto) physicochemical properties
        df_with_threshold = apply_thresholds(df_gen_props=df_gen_props, property_thresholds=[qed_t, sascore_t, tanimoto_t], mw=mw, ta_direction=ta_direction)
        print('SECOND FILTER:')
        print(len(df_with_threshold))
        
        if len(df_with_threshold)==0:
            raise Exception("No valid compounds generated after the second filter.")
        
        # Apply 3rd filter: FilterCatog PAINS, Brenk, NIH, CHEMBL (optional)
        if skip_third_filter:
            print('THIRD FILTER: skipped by flag (--skip_catalog_filter)')
            df_filtered = df_with_threshold
        else:
            df_filtered = apply_unwanted_substruct_filters(df_with_threshold)
            print('THIRD FILTER:')
            print(len(df_filtered))
        
        if len(df_filtered)==0:
            raise Exception("No valid compounds generated after the last filter.")
        
        # Apply additional filter if indicated: Trifurcated substructure filter
        print('ADDITIONAL FILTERS:')
        if trifurcated:
            df_filtered = apply_trifurcated_filter(df_filtered, min_branch_size=trifurcated)
            print(f'  - Compounds after trifurcated filter (min branch size {trifurcated}): ', len(df_filtered))
        
        if len(df_filtered)==0:
            raise Exception("No valid compounds generated after the trifurcated filter.")
        
        timef = time.time() - timef0
        print('Time Filters Chemical AL: ', timef, 'seconds')
        print('Time human readable:', time.strftime("%H:%M:%S", time.gmtime(timef)))

        if not os.path.exists(inner_output_dir + '_' + str(i+1)):
            os.mkdir(inner_output_dir + '_' + str(i+1))
            
        # Generate a CSV with the filtered compounds
        df_filtered.to_csv(inner_output_dir + '_' + str(i+1) + '/' + name + '_generated_smiles_threshold_' + str(i) + '.csv')
        
        # Merge the generated compounds with the compounds from the specific training set
        df_init_props, df_filtered = filter_unique_molecules(df_init_props, df_filtered) # remove duplicates
        df_merge_rounds = pd.concat([df_filtered['smiles'], df_init_props['smiles']])
        df_merge_rounds.reset_index(inplace=True, drop=True)
        
        df_merge_rounds.to_csv(inner_output_dir + '_' + str(i+1) + '/' + name + '_generated_smiles_merged.csv', index=False, header=None)
        
        # Change the name of the specific training set to the new set (generated + previous specific training set)
        specific_smiles = inner_output_dir + '_' + str(i+1) + '/' + name + '_generated_smiles_merged.csv'
        
        timeal = time.time() - timeal0
        print('Time Chemical AL cycle: ', timeal, 'seconds')
        print('Time human readable:', time.strftime("%H:%M:%S", time.gmtime(timeal)))

    # After all active-learning cycles, report total runtime
    time_total = time.time() - time0
    print(f"\nTotal execution time: {time_total:.2f} seconds")
    print("Total human-readable time:", time.strftime("%H:%M:%S", time.gmtime(time_total)))

    # Persist execution time in the final report directory
    final_report_dir = inner_output_dir + '_' + str(inner_al)
    if os.path.exists(final_report_dir):
        write_report(train_size=original_train_size, val_size=original_val_size, test_size=original_test_size, test_loss=score,
                     test_acc=acc, epochs=num_epochs, batch_size=batch_size, sampling_temp=sampling_temp, top_p=top_p,
                     quantity_requested=quant, valid_generated=valid_generated,
                     total_attempts=total_attempts, not_valid=not_valid, name=name, outdir=final_report_dir,
                     execution_time=time_total, seed=seed_value)

if __name__ == "__main__":
    # try:
    #     main()
    # except Exception as e:
    #     print("Shutting down with error: ", e) 
    main()
