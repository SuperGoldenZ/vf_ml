from dataclasses import dataclass

@dataclass
class FrameData:
    """Class for holding data of a frame in Virtua Fighter"""
    stage: str
    p1_health: int
    p2_health: int
    p1_rank: int
    p2_rank: int
    p1_drinks: int
    p2_drinks: int
    p1_ringname: str
    p2_ringname: str
    p1_rounds_won_so_far: int
    p2_rounds_won_so_far: int
    time_remaining: float
    
    