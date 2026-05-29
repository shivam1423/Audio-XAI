#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import get_model_predictions


def explain_all(data_loader, explainer, model, preprocessor=None):
    """
    Generate explanations for all images/audio in the dataset.
    
    Args:
        data_loader: DataLoader for images or audio files
        explainer: RISE or other explainer instance
        model: Model for predictions
        preprocessor: Optional preprocessor for audio files
    """
    target = np.empty(len(data_loader), np.int64)
    filenames = []
    
    # Get predictions for all images/audio
    for i, (input_data, _, path) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Predicting labels')):
        if preprocessor is not None:
            # Process audio file
            model_input = preprocessor(input_data[0]).cuda()
        else:
            # Use pre-generated spectrogram/image
            model_input = input_data.cuda()
            
        p, c = get_model_predictions(model, model_input)
        target[i] = c[0]
        filenames.append(path[0])

    # Generate explanations
    explanations = np.empty((len(data_loader), *explainer.input_size))
    for i, (input_data, _, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Explaining images')):
        if preprocessor is not None:
            # Process audio file
            model_input = preprocessor(input_data[0]).cuda()
        else:
            # Use pre-generated spectrogram/image
            model_input = input_data.cuda()
            
        saliency_maps = explainer(model_input)
        explanations[i] = saliency_maps[target[i]].cpu().numpy()
    
    return explanations, target, filenames


def save_saliency_results(explanations, targets, filenames, output_dir, results_info):
    """Save saliency results to files."""
    import pickle
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy array
    np.save(os.path.join(output_dir, 'all_saliency_maps.npy'), explanations)
    
    # Save metadata
    results = {
        'explanations': explanations,
        'targets': targets,
        'filenames': filenames,
        'dataset_size': len(explanations),
        'input_size': results_info['input_size'],
        'model': 'resnet50',
        'mask_type': results_info['mask_type'],
        'soft_masking': results_info['soft_masking'],
        'edge_sigma_px': results_info['edge_sigma_px'],
        'occlusion': results_info['occlusion'],
        'mask_fractions': results_info['mask_fractions'],
        'N': results_info['N'],
        'mask_file': results_info['mask_file'],
        'occlusion_method': results_info['occlusion']
    }
    
    with open(os.path.join(output_dir, 'saliency_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved saliency maps for {len(explanations)} images to {output_dir}/")
    return results
