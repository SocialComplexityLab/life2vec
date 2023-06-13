from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.seed import reset_seed
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.distributed import init_dist_connection

from pathlib import Path

class DDPPluginFileInit(DDPPlugin):
    """
    Custom DDP strategy that uses file:///~/torch_init.tmp as  
    init_method argument. This resolves issues for training with 
     >2 devices.
    """
    def setup_environment(self) -> None:
        
        init_file = Path.home() / "torch_init.tmp"

        # start the other scripts
        if not self.cluster_environment.creates_processes_externally:
            
            # First remove file if exists
            if init_file.exists():
                init_file.unlink()

            self._call_children_scripts()

        # set the task idx
        self.task_idx = self.cluster_environment.local_rank()

        # Format init_method argument
        init_method = f"file:///{init_file.as_posix()}"
        self.setup_distributed(init_method)


    def setup_distributed(self, init_method):
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        init_dist_connection(
            self.cluster_environment, 
            self.torch_distributed_backend,
            init_method=init_method,
            )
