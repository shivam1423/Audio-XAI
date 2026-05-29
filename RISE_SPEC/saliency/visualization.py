#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import get_model_predictions
from rise_utils import tensor_imshow, get_class_name


def save_individual_saliency_maps(data_loader, explanations, filenames, output_dir, datadir, model, preprocessor=None):
    """
    Save individual saliency maps and visualizations.
    
    Args:
        data_loader: DataLoader for images or audio files
        explanations: Saliency map explanations
        filenames: List of filenames
        output_dir: Output directory
        datadir: Data directory for relative path calculation
        model: Model for predictions
        preprocessor: Optional preprocessor for audio files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (input_data, _, path) in enumerate(tqdm(data_loader, desc='Saving individual maps')):
        if preprocessor is not None:
            # Process audio file
            model_input = preprocessor(input_data[0]).cuda()
        else:
            # Use pre-generated spectrogram/image
            model_input = input_data.cuda()
            
        p, c = get_model_predictions(model, model_input)
        p, c = p[0].item(), c[0].item()

        original_path = path[0]
        original_name = os.path.basename(original_path)
        name_without_ext = os.path.splitext(original_name)[0]

        rel_path = os.path.relpath(original_path, datadir)
        rel_dir = os.path.dirname(rel_path)
        
        if rel_dir:
            subdir_output = os.path.join(output_dir, rel_dir)
            os.makedirs(subdir_output, exist_ok=True)
            output_path = subdir_output
        else:
            output_path = output_dir

        saliency_map = explanations[i]
        np.save(os.path.join(output_path, f'{name_without_ext}_saliency.npy'), saliency_map)

        # Create visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.axis('off')
        plt.title(f'{100 * p:.2f}% {get_class_name(c)}')
        if preprocessor is not None:
            tensor_imshow(model_input[0])  # Show preprocessed spectrogram
        else:
            tensor_imshow(input_data[0])
        plt.subplot(122)
        plt.axis('off')
        plt.title(get_class_name(c))
        if preprocessor is not None:
            tensor_imshow(model_input[0])
        else:
            tensor_imshow(input_data[0])
        plt.imshow(saliency_map, cmap='jet', alpha=0.5)
        plt.savefig(os.path.join(output_path, f'{name_without_ext}_saliency.png'), bbox_inches='tight', dpi=150)
        plt.close()
