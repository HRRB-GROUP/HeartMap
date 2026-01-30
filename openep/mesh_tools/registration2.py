from copy import deepcopy
from ..data_structures.case import Case


class CPDReconstructedCase:
    def __init__(self, source_case: Case):
        self.source_case = source_case

    def build(self, registered_points, name_suffix="_registered"):
        if registered_points.shape != self.source_case.points.shape:
            raise ValueError("Registered points shape mismatch")

        return Case(
            name=self.source_case.name + name_suffix,
            points=registered_points,
            indices=self.source_case.indices.copy(),
            fields=deepcopy(self.source_case.fields),
            electric=deepcopy(self.source_case.electric),
            ablation=deepcopy(self.source_case.ablation),
            notes=deepcopy(self.source_case.notes),
            vectors=deepcopy(self.source_case.vectors),
        )
