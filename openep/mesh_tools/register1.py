import numpy as np
import pycpd
import time


class CPDRegistrationPoints:
    def __init__(self, method="affine", kwargs=None):
        self.method = method
        self.kwargs = kwargs or {}
        self.history = []

    def _build_registrator(self, source, target):
        if self.method == "rigid":
            return pycpd.RigidRegistration(X=target, Y=source, **self.kwargs)
        elif self.method == "affine":
            return pycpd.AffineRegistration(X=target, Y=source, **self.kwargs)
        elif self.method == "deformable":
            if "source_id" in self.kwargs:
                return pycpd.ConstrainedDeformableRegistration(
                    X=target, Y=source, **self.kwargs
                )
            return pycpd.DeformableRegistration(X=target, Y=source, **self.kwargs)
        else:
            raise ValueError(f"Unknown CPD method: {self.method}")

    def _callback(self, iteration, error, X, Y):
        self.history.append(Y.copy())
        print(f"Itération {iteration}, Erreur: {error:.4f}")

    def run(self, source_points: np.ndarray, target_points: np.ndarray):
        reg = self._build_registrator(source_points, target_points)

        start = time.time()
        TY, _ = reg.register(callback=self._callback)
        elapsed = time.time() - start

        print(f"CPD finished in {elapsed:.2f}s")
        return TY
