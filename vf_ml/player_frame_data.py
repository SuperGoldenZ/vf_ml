from dataclasses import dataclass


@dataclass
class PlayerFrameData:
    """Class for holding data of a frame in Virtua Fighter"""

    health: int
    rank: int
    drinks: int
    ringname: str
    character: str
    rounds_won_so_far: int
