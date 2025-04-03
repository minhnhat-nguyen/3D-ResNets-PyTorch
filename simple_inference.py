import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from dataclasses import dataclass

from model import generate_model

UCF101_CLASSES = [
    'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling',
    'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball',
    'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair',
    'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag',
    'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk',
    'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen',
    'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics',
    'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering',
    'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage',
    'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing',
    'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking',
    'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing',
    'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing',
    'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute',
    'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla',
    'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch',
    'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing',
    'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing',
    'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings',
    'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi',
    'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing',
    'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups',
    'WritingOnBoard', 'YoYo'
]

@dataclass
class Options:
    model: str = 'wideresnet'
    model_depth: int = 50
    n_classes: int = 101
    n_input_channels: int = 3
    wide_resnet_k: int = 2
    resnet_shortcut: str = 'B'
    conv1_t_size: int = 7
    conv1_t_stride: int = 1
    no_max_pool: bool = False
    sample_size: int = 112
    sample_duration: int = 16

def extract_frames_smart_sampling(video_path, sample_duration=16, clips_per_second=1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps if fps > 0 else 0
    print(f"Video has {total_frames} frames at {fps} FPS (duration: {duration_sec:.2f} seconds)")
    
    target_clips = max(1, int(duration_sec * clips_per_second))
    print(f"Target number of clips: {target_clips}")
    
    if total_frames <= sample_duration:
        clips = [[]]
        for frame_idx in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clips[0].append(frame)
            else:
                print(f"Warning: Failed to read frame at index {frame_idx}")
        while len(clips[0]) < sample_duration:
            clips[0].append(clips[0][-1].copy())
    else:
        clips = []
        for clip_idx in range(target_clips):
            start_frame = int(clip_idx * (total_frames / target_clips))
            end_frame = min(start_frame + sample_duration, total_frames)
            sample_indices = np.linspace(start_frame, end_frame - 1, sample_duration, dtype=int)
            print(f"Clip {clip_idx+1}: frames {start_frame} to {end_frame - 1}")
            frames = []
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    print(f"Warning: Failed to read frame at index {frame_idx}")
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                        else:
                            raise ValueError(f"Failed to read any frames from video")
            clips.append(frames)
    cap.release()
    print(f"Successfully extracted {len(clips)} clips")
    return clips

def preprocess_clips(clips, sample_size=112):
    transform = transforms.Compose([
        transforms.Resize(sample_size),
        transforms.CenterCrop(sample_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989]
        )
    ])
    processed_clips = []
    for frames in clips:
        processed_frames = [transform(Image.fromarray(frame)) for frame in frames]
        clip_tensor = torch.stack(processed_frames, 1)
        processed_clips.append(clip_tensor)
    return processed_clips

def load_model(opt, checkpoint_path):
    model = generate_model(opt)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, device

def predict_clips(model, clips, device, class_names, top_k=5):
    all_predictions = []
    all_scores = torch.zeros(len(class_names))
    for clip_idx, clip in enumerate(clips):
        clip_tensor = clip.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(clip_tensor)
            outputs = F.softmax(outputs, dim=1).cpu().squeeze()
        all_scores += outputs
        sorted_scores, locs = torch.topk(outputs, k=min(top_k, len(class_names)))
        clip_predictions = [
            {'label': class_names[locs[i].item()], 'score': sorted_scores[i].item()}
            for i in range(sorted_scores.size(0))
        ]
        all_predictions.append(clip_predictions)
        print(f"Clip {clip_idx+1} top prediction: {clip_predictions[0]['label']} ({clip_predictions[0]['score']:.4f})")
    all_scores /= len(clips)
    sorted_scores, locs = torch.topk(all_scores, k=min(top_k, len(class_names)))
    final_predictions = [
        {'label': class_names[locs[i].item()], 'score': sorted_scores[i].item()}
        for i in range(sorted_scores.size(0))
    ]
    return final_predictions, all_predictions

def main():
    parser = argparse.ArgumentParser(description="Simple video action prediction")
    parser.add_argument("--video", required=True, type=Path, help="Path to input video file")
    parser.add_argument("--checkpoint", default="saved/save_120.pth", type=Path, 
                       help="Path to checkpoint file")
    parser.add_argument("--top_k", type=int, default=5, 
                       help="Number of top predictions to show")
    parser.add_argument("--clips_per_second", type=float, default=1.0,
                       help="Number of clips to extract per second of video")
    
    args = parser.parse_args()
    
    try:
        if not args.video.exists():
            raise FileNotFoundError(f"Video file not found: {args.video}")
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        
        opt = Options()
        
        model, device = load_model(opt, args.checkpoint)
        
        print(f"Extracting clips from video: {args.video}")
        clips = extract_frames_smart_sampling(
            args.video, 
            opt.sample_duration,
            args.clips_per_second
        )
        
        print("Preprocessing clips...")
        processed_clips = preprocess_clips(clips, opt.sample_size)
        
        print("Running prediction...")
        final_predictions, clip_predictions = predict_clips(
            model, processed_clips, device, UCF101_CLASSES, args.top_k
        )
        
        print("\nPredictions for each clip:")
        for clip_idx, preds in enumerate(clip_predictions):
            print(f"\nClip {clip_idx+1}:")
            for i, pred in enumerate(preds):
                print(f"  {i+1}. {pred['label']} ({pred['score']:.4f})")
        
        print("\nFinal aggregated predictions:")
        for i, pred in enumerate(final_predictions):
            print(f"  {i+1}. {pred['label']} ({pred['score']:.4f})")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()