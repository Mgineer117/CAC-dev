"""CARL_M: CARL with the Mahalanobis tracking reward  -||e||^2_M.

Identical to CARL in every respect except:
  - reward_mode is always "mahal" (never "inverse"), so get_rewards() uses
    the raw -tracking_scaler * e^T M e reward without the 1/(1+|r|) wrapper.
  - self.name = "CARL_M" for clean WandB / log namespacing.

All training logic (CMG warmup, PPO actor, joint W update) is inherited
unchanged from CARL.
"""

from policy.carl import CARL


class CARL_M(CARL):
    """CARL variant that uses -||e||^2_M as the tracking reward."""

    def __init__(self, *args, **kwargs):
        # Force reward_mode to "mahal" regardless of what was passed in.
        # Any value that is not "inverse" activates the Mahalanobis path in
        # CARL.get_rewards(), so this is the minimal override needed.
        kwargs["reward_mode"] = "mahal"
        super().__init__(*args, **kwargs)
        # Override the name so WandB metrics are namespaced separately from CARL.
        self.name = "CARL_M"
