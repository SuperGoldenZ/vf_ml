from vf_ml import player_frame_data
from dataclasses import dataclass

@dataclass
class FrameData:
    """Class for holding data of a frame in Virtua Fighter"""
    stage: str
    time_remaining: float
    p1_frame_data: player_frame_data.PlayerFrameData
    p2_frame_data: player_frame_data.PlayerFrameData
    
    