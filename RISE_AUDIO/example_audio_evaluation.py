# Create a separate evaluation script
from src.models.resnet50 import ResNetModel
from src.evaluation import evaluate_audio_saliency_maps

model = ResNetModel()
evaluate_audio_saliency_maps(
    audio_dir="../ESC50/audio_esc10",
    saliency_dir="results/saliency/test_48000_SR_all_multiple_6000masks_time",
    output_dir="results/saliency/test_48000_SR_all_multiple_6000masks_time/evaluation1",
    # audio_dir="test_audio",
    # saliency_dir="results/saliency/test_48000_SR_all_single_6000masks",
    # output_dir="results/audio_evaluation/test48000_SR_all_single_6000masks",
    model=model,
    steps=224,
    verbose=1,
    # ins_dir="results/insertion_images",
    # del_dir="results/deletion_images"
)