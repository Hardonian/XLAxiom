"""Google Cloud Accelerated Processing Kit (xpk) and GKE TPU cluster spec generator.

Generates workload creation commands and declarative specifications for running
XLAxiom campaigns on Google Kubernetes Engine (GKE) TPU clusters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class XPKWorkloadSpec:
    workload_name: str
    cluster_name: str
    zone: str
    device_type: str
    num_slices: int
    docker_image: str
    command: str

    def to_cli_command(self) -> str:
        """Format as standard 'xpk workload create' invocation."""
        return (
            f"xpk workload create \\\n"
            f"  --workload={self.workload_name} \\\n"
            f"  --cluster={self.cluster_name} \\\n"
            f"  --zone={self.zone} \\\n"
            f"  --device-type={self.device_type} \\\n"
            f"  --num-slices={self.num_slices} \\\n"
            f"  --docker-image={self.docker_image} \\\n"
            f"  --command={self.command!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_xpk_workload_spec(
    workload_name: str = "xlaxiom-tpu-eval",
    cluster_name: str = "tpu-cluster-v5e",
    zone: str = "us-south1-a",
    device_type: str = "v5litepod-4",
    num_slices: int = 1,
    docker_image: str = "us-docker.pkg.dev/google-cloud/xlaxiom/runner:latest",
    command: str = "xlaxiom property campaign --backend jax-tpu-jit --cases 100",
) -> XPKWorkloadSpec:
    """Generate xpk workload configuration for GKE TPU orchestrations."""
    return XPKWorkloadSpec(
        workload_name=workload_name,
        cluster_name=cluster_name,
        zone=zone,
        device_type=device_type,
        num_slices=num_slices,
        docker_image=docker_image,
        command=command,
    )

