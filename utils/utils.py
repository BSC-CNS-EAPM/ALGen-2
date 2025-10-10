import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Lipinski, QED, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem import AllChem, DataStructs
from . import sascorer

import matplotlib.pyplot as plt
import seaborn as sns

import selfies as sf
import deepsmiles as ds

from collections import Counter


def write_report(train_size, val_size, test_size, test_loss, test_acc, epochs, batch_size, sampling_temp, quant, not_valid, name, outdir):
    with open(outdir + "/report_" + name + '.out', 'w+') as r:
        r.write('#######################################\n')
        r.write('Report for generative run ' + name + '\n')
        r.write('#######################################\n')
        r.write('Model trained for: ' + str(epochs)+ ' epoch' + '\n')
        r.write('Chosen batch size: ' + str(batch_size)+ '\n')
        r.write('Sampling temperature: ' + str(sampling_temp)+ '\n')
        r.write('#######################################'+ '\n')
        r.write('Train size: ' + str(train_size)+ '\n')
        r.write('Validation size: ' + str(val_size)+ '\n')
        r.write('Test size: ' + str(test_size)+ '\n')
        r.write('#######################################'+ '\n')
        r.write('Test loss: ' + str(test_loss)+ '\n')
        r.write('Test accuracy: ' + str(test_acc)+ '\n')
        r.write('#######################################'+ '\n')
        r.write('Tried to generate ' + str(quant) + ' compounds and ' + str(not_valid) + ' were not valid'+ '\n')
        if not_valid > 0:
            success_rate = (1 - float(not_valid)/(float(quant))) * 100
        else:
            success_rate = 0.0
        r.write('Success rate: ' + str(success_rate) + '%')

def to_gpu():
    try:
        print('Num GPUs available {}'.format(tf.config.experimental.list_physical_devices('GPU')))

    except:
        print('Tensorflow did not find any NVIDIA GPU or Tensorflow-GPU is not installed.')

def plot_acc_loss(history, name, output_dir):
    fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

    axis1.plot(history.history["acc"], label='Train', linewidth=3)
    axis1.plot(history.history["val_acc"], label='Validation', linewidth=3)
    axis1.set_title('Model accuracy', fontsize=16, color="white")
    axis1.set_ylabel('accuracy')
    axis1.set_xlabel('epoch')
    axis1.legend(loc='lower right')

    axis2.plot(history.history["loss"], label='Train', linewidth=3)
    axis2.plot(history.
history["val_loss"], label='Validation', linewidth=3)
    axis2.set_title('Model loss', fontsize=16, color="white")
    axis2.set_ylabel('loss')
    axis2.set_xlabel('epoch')
    axis2.legend(loc='upper right')

    plt.savefig(output_dir + '/' + name + "_accuracy_loss.png", format='png')

    return 0


def sample_with_temp(preds, sampling_temp):

    streched = np.log(preds) / sampling_temp
    streched_probs = np.exp(streched) / np.sum(np.exp(streched))

    return np.random.choice(range(len(streched)), p=streched_probs)


def compute_tanimoto_init(specific_set):
    '''
    '''
    
    bulk = []
    fps = []

    with open(specific_set, 'r') as f:
        for i in f:
            bulk.append(Chem.MolFromSmiles(i))
    
    for x in bulk:
        try:
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4)
            fps.append(fpgen.GetSparseCountFingerprint(x))
        except:
            pass

    return fps


def tanimoto2(compound, specific_set_fps, ta_direction):
    '''
    '''
    comp_mol = Chem.MolFromSmiles(compound)

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4)
    comp_fps = fpgen.GetSparseCountFingerprint(comp_mol)

    sims = DataStructs.BulkTanimotoSimilarity(comp_fps, specific_set_fps)
    
    if ta_direction == '<':
        return np.mean(sims), np.max(sims)
    elif ta_direction == '>':
        return np.mean(sims), np.max(sims), sims
    else:
        raise ValueError('Indicate the ta_direction: < or >')

def compute_properties(smi):
    '''
    '''
    print('smiles: ', smi)
    mol = Chem.MolFromSmiles(smi)

    sa = sascorer.calculateScore(mol)
    qed = round(QED.weights_max(mol), 2)
    mw = Descriptors.ExactMolWt(mol)
    logp = round(Descriptors.MolLogP(mol), 2)
    tpsa = round(Descriptors.TPSA(mol), 2)
    nhd = Lipinski.NumHDonors(mol)
    nha = Lipinski.NumHAcceptors(mol)

    return smi, sa, qed, mw, logp, tpsa, nhd, nha

def apply_thresholds(df_gen_props, property_thresholds, mw=None, ta_direction='<'):
    '''
    '''
    
    df_gen_props_t = df_gen_props[df_gen_props['QED'] >= property_thresholds[0]]
    print('  - Compounds after QED:', len(df_gen_props_t))
    df_gen_props_t = df_gen_props_t[df_gen_props_t['SAscore'] <= property_thresholds[1]]
    print('  - Compounds after SAscore:', len(df_gen_props_t))
    if ta_direction == '<':
        df_gen_props_t = df_gen_props_t[df_gen_props_t['tan_max'] <= property_thresholds[2]]
    elif ta_direction == '>':
        # if ta threshold value (or higher) is inside the sims list, keep the molecule
        df_gen_props_t = df_gen_props_t[df_gen_props_t['sims'].apply(lambda x: any(i >= property_thresholds[2] for i in x))]
    else:
        raise ValueError('Indicate the ta_direction: < or >')
    print('  - Compounds after tanimoto:', len(df_gen_props_t))
    if mw:
        df_gen_props_t = df_gen_props_t[(df_gen_props_t['mw'] >= mw[0]) & (df_gen_props_t['mw'] <= mw[1])]
    print('  - Compounds after mw filter:', len(df_gen_props_t))
    
    df_gen_props_t = df_gen_props_t[['smiles', 'SAscore','QED', 'mw', 'logp', 'tpsa', 'nHD', 'nHA', 'tan_mean', 'tan_max']]
    return df_gen_props_t

def histogram_properties(df_initial, df_generated, property, out_dir):
    '''
    '''

    plt.figure(figsize=(15,10))
    sns.histplot(data=df_initial, x=property, color='gray', kde=True)
    sns.histplot(data=df_generated, x=property, color='lightgreen', kde=True)

    plt.title('{} Distributions'.format(property))
    plt.suptitle('Run: {}'.format(out_dir.split('/')[-1]))
    plt.legend(['Initial', 'Generated'])

    plt.savefig('{}/{}_{}_hist.png'.format(out_dir, out_dir.format('/')[-1], property), format='png')

def smiles2selfies(smiles_file):
    '''
    '''

    selfies_set = {}

    with open(smiles_file, 'r') as f:
        for smile in f:
            selfies_set[smile.strip()] = sf.encoder(smile.strip())
    
    char_alphabet =  sf.get_alphabet_from_selfies(selfies_set.values())
    char_alphabet.add("[nop]")
    char_alphabet = list(sorted(char_alphabet)) 

    return selfies_set, char_alphabet

def selfies_one_hot_encoder(selfies_set, char_alphabet):
    '''
    '''
    discard = 0

    one_hot_selfies = []
    labels = []

    pad_to_len = max(sf.len_selfies(s) for s in selfies_set)  # 5
    symbol_to_idx = {s: i for i, s in enumerate(char_alphabet)}

    for selfie in selfies_set:
        try:
            label, one_hot = sf.selfies_to_encoding(
            selfies=selfie,
            vocab_stoi=symbol_to_idx,
            pad_to_len=pad_to_len,
            enc_type="both"
            )

            one_hot_selfies.append(one_hot)
            labels.append(label)
        except:
            discard += 1
        
    print(discard)

    return np.array(one_hot_selfies)[:,0:-1,:], labels, np.array(one_hot_selfies)[:,1:,:]
    
def selfies_one_hot_decoder(one_hot_selfies, alphabet_char):
    '''
    '''
    
    selfies = []

    for one_hot_encoding in one_hot_selfies:
        selfie = sf.encoding_to_selfies(encoding=one_hot_encoding,
        vocab_itos=alphabet_char,
        enc_type='both'
        )

        selfies.append(selfie)
    
    return selfies

def remove_duplicates_tanimoto(set1, set2, threshold=1):
    """Remove duplicates molecules from set2 based on tanimoto similarity to
       to molecules in set1."""
    
    fingerprints_set1 = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in set1]

    # Compute similarity and apply the threshold
    unique_molecules_set2 = []
    for mol in set2:
        fp_mol = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        similarities = DataStructs.BulkTanimotoSimilarity(fp_mol, fingerprints_set1)
        
        if not any(sim >= threshold for sim in similarities):
            unique_molecules_set2.append(mol)
    
    return unique_molecules_set2
    
def filter_unique_molecules(df1, df2, threshold=1):
    """Remove duplicates from the second dataframe it they already existed in
       the first df. It is applied when merging dfs between inners."""
       
    set1 = [Chem.MolFromSmiles(smi) for smi in df1['smiles']]
    set2 = [Chem.MolFromSmiles(smi) for smi in df2['smiles']]
    
    # Ensure no NoneType molecules (invalid SMILES)
    set1 = [mol for mol in set1 if mol is not None]
    set2_valid = [(mol, idx) for idx, mol in enumerate(set2) if mol is not None]
    
    # Apply remove duplicates
    set2_mols, set2_indices = zip(*set2_valid) if set2_valid else ([], [])
    unique_molecules_set2 = remove_duplicates_tanimoto(set1, set2_mols, threshold=threshold)
    
    # Get the indices of unique molecules and filter the second dataframe
    unique_indices = [set2_indices[i] for i, mol in enumerate(set2_mols) if mol in unique_molecules_set2]
    filtered_df2 = df2.iloc[unique_indices].reset_index(drop=True)
    
    return df1, filtered_df2

def filter_rings_8_atoms(mol):
    """Filter the molecules that have rings with 8 or more atoms"""
    rings = mol.GetRingInfo().AtomRings()
    #print(mol.GetRingInfo().NumRings())
    for ring in rings:        
        if len(ring) >= 8:
            print('Ring with 8 or more atoms found on molecule with SMILES:', Chem.MolToSmiles(mol))
            return None
    return mol

def filter_3_more_fused_rings(mol):
    """Filter the molecules that have 3 or more fused rings"""
    rings = mol.GetRingInfo().AtomRings()
    # Flatten the list and count occurrences, then get items with count > 1
    fused_rings_idxs =  [idx for idx, count in Counter([i for t in rings for i in t]).items() if count > 1]
    if len(fused_rings_idxs) >= 4:
        # Check if these idxs are in the same ring to be considered all fused together
        intersected_rings = []
        for i, ring1 in enumerate(rings):
            for j, ring2 in enumerate(rings):
                if i < j:
                    intersection =  set(ring1).intersection(set(ring2))
                    if len(intersection) > 1:
                        intersected_rings.append([i, j])
        intersected_rings_idxs = [i for t in intersected_rings for i in t]
        unique = set(intersected_rings_idxs)
        if len(unique) < len(intersected_rings_idxs):
            print('3 or more fused rings found on molecule with SMILES:', Chem.MolToSmiles(mol))
            return None
        else:
            return mol
    else:
        return mol

def filter_3_4_rings_fused(mol):
    """Filter the molecules that have 3 or 4-atom rings fused with other rings"""
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        if len(ring) == 3 or len(ring) == 4:
            other_rings = [r for r in rings if r != ring]
            other_rings = set([i for t in other_rings for i in t])
            fused_rings_idxs = set(ring).intersection(other_rings)
            if len(fused_rings_idxs) > 1:
                print('Fused 3-4-atom rings found on molecule with SMILES:', Chem.MolToSmiles(mol))
                return None
    return mol

def filter_4_ring_with_substitutions(mol):
    """Filter the molecules that have 4-atom rings with more than 2 substitutions"""
    if not mol: return None
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        if len(ring) == 4:
            atoms = [mol.GetAtomWithIdx(i) for i in ring]
            bonds = [len(atom.GetNeighbors()) for atom in atoms]
            if sum([1 for b in bonds if b > 2]) > 2:
                print('4-atom ring with more than 2 substitutions found on molecule with SMILES:', Chem.MolToSmiles(mol))
                return None
    return mol

def filter_more_than_2_7rings(mol):
    """Filter the molecules that have 2 or more 7-membered rings"""
    rings = mol.GetRingInfo().AtomRings()
    count = 0
    for ring in rings:
        if len(ring) >= 7:
            count += 1
    if count >= 2:
        print('2 or more 7-membered rings found on molecules with SMILES:', Chem.MolToSmiles(mol))
        return None
    return mol

def filter_consecutive_nitrogens_ring(mol):
    """Filter the molecules that have consecutive nitrogens in a ring"""
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in ring]
        if 'N' not in atoms: continue
        for i in range(len(ring)-1):
            if mol.GetAtomWithIdx(ring[i]).GetAtomicNum() == 7 and mol.GetAtomWithIdx(ring[i+1]).GetAtomicNum() == 7:
                print('Consecutive nitrogens in a ring found on molecule with SMILES:', Chem.MolToSmiles(mol))
                return None
        if mol.GetAtomWithIdx(ring[0]).GetAtomicNum() == 7 and mol.GetAtomWithIdx(ring[-1]).GetAtomicNum() == 7:
            print('Consecutive nitrogens in a ring found on molecule with SMILES:', Chem.MolToSmiles(mol))
            return None
    return mol

def filter_more_than_4_chiral_centers(mol):
    """Filter the molecules that have more than 4 chiral centers"""
    chiral = Chem.FindMolChiralCenters(mol)
    if len(chiral) > 4:
        print('More than 4 chiral centers found on molecule with SMILES:', Chem.MolToSmiles(mol))
        return None
    return mol

def filter_manual_SMARTS_pattens(mol):
    """Filter the molecules that have manual SMARTS patterns"""
    patterns = ['[C-]']
    for pattern in patterns:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
            print('Manual SMARTS pattern found on molecule with SMILES:', Chem.MolToSmiles(mol), ' --> ', pattern)
            return None
    return mol

def filter_PAINS(mol):
    params_pains = FilterCatalogParams()
    params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog_pains = FilterCatalog(params_pains)
    flag = catalog_pains.HasMatch(mol) # Check if there is a matching PAINS
    if flag:
        description = [entry.GetDescription() for entry in catalog_pains.GetMatches(mol)]
        print('PAINS found on molecule with SMILES:', Chem.MolToSmiles(mol), ' --> ', description)
        return None
    else:
        return mol

def filter_Brenk(mol):
    params_unwanted = FilterCatalogParams()
    params_unwanted.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog_unwanted = FilterCatalog(params_unwanted)
    flag = catalog_unwanted.HasMatch(mol) # Check if there is a matching unwanted substructure
    if flag:
        description = [entry.GetDescription() for entry in catalog_unwanted.GetMatches(mol)]
        print('Brenk found on molecule with SMILES:', Chem.MolToSmiles(mol), ' --> ', description)
        return None
    else:
        return mol

def filter_NIH(mol):
    params_nih = FilterCatalogParams()
    params_nih.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
    catalog_nih = FilterCatalog(params_nih)
    flag = catalog_nih.HasMatch(mol) # Check if there is a matching NIH
    if flag:
        description = [entry.GetDescription() for entry in catalog_nih.GetMatches(mol)]
        print('NIH found on molecule with SMILES:', Chem.MolToSmiles(mol), ' --> ', description)
        return None
    else:
        return mol
    
def filter_REOS(mol):
    """Adds Pat Walter's Chembl Filters.
        The filters are based on the REOS (Rapid Elimination Of Swill).
        The Catalog CHEMBL contains all these filters:
        CHEMBL = CHEMBL_Glaxo | CHEMBL_Dundee | CHEMBL_BMS | CHEMBL_SureChEMBL |
        CHEMBL_MLSMR | CHEMBL_Inpharmatica | CHEMBL_LINT"""
        
    params_reos = FilterCatalogParams()
    params_reos.AddCatalog(FilterCatalogParams.FilterCatalogs.CHEMBL)
    catalog_reos = FilterCatalog(params_reos)
    flag = catalog_reos.HasMatch(mol) # Check if there is a matching REOS
    if flag:
        description = [entry.GetDescription() for entry in catalog_reos.GetMatches(mol)]
        print('REOS found on molecule with SMILES:', Chem.MolToSmiles(mol), ' --> ', description)
        return None
    else:
        return mol

def apply_structural_filters(df):
    """Apply structural filters of unwanted ring substructures"""

    df.dropna(subset=['mols'], inplace=True)  
    df['mols'] = df['mols'].apply(filter_rings_8_atoms)
    df.dropna(subset=['mols'], inplace=True)
    df['mols'] = df['mols'].apply(filter_3_more_fused_rings)
    df.dropna(subset=['mols'], inplace=True)
    df['mols'] = df['mols'].apply(filter_3_4_rings_fused)
    df.dropna(subset=['mols'], inplace=True)
    df['mols'] = df['mols'].apply(filter_4_ring_with_substitutions)
    df.dropna(subset=['mols'], inplace=True)
    df['mols'] = df['mols'].apply(filter_more_than_2_7rings)
    df.dropna(subset=['mols'], inplace=True)
    df['mols'] = df['mols'].apply(filter_consecutive_nitrogens_ring)
    df.dropna(subset=['mols'], inplace=True)
    df['mols'] = df['mols'].apply(filter_more_than_4_chiral_centers)
    df.dropna(subset=['mols'], inplace=True)
    df['mols'] = df['mols'].apply(filter_manual_SMARTS_pattens)
    df.dropna(subset=['mols'], inplace=True)
    
    return df

def apply_unwanted_substruct_filters(df):
    """Apply structural filters of PAINS, Brenk, and NIH"""

    df['mols'] = df['smiles'].apply(lambda smile: Chem.MolFromSmiles(smile) if pd.notna(smile) else None)
    df.dropna(subset=['mols'], inplace=True)
    df['mols'] = df['mols'].apply(filter_PAINS)
    df.dropna(subset=['mols'], inplace=True)
    df['mols'] = df['mols'].apply(filter_Brenk)
    df.dropna(subset=['mols'], inplace=True)
    df['mols'] = df['mols'].apply(filter_NIH)
    df.dropna(subset=['mols'], inplace=True)
    df['mols'] = df['mols'].apply(filter_REOS)
    df.dropna(subset=['mols'], inplace=True)

    return df
