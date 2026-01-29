import numpy as np
import pycpd
import time

class CPDRegistrationWorker:
    def __init__(self, pars):
        self.pars = pars['pars']
        self.history = []

    def _setup_registration(self, source, target):
        method = self.pars.get('method', 'affine')
        kwargs = self.pars.get('kwargs', {})
        
        # Mapping des classes
        classes = {
            'rigid': pycpd.RigidRegistration,
            'affine': pycpd.AffineRegistration,
            'deformable': pycpd.DeformableRegistration
        }
        
        # Cas spécial pour constrained deformable
        if method == 'deformable' and 'source_id' in kwargs:
            return pycpd.ConstrainedDeformableRegistration(X=target, Y=source, **kwargs)
        
        if method not in classes:
            raise ValueError(f"Méthode inconnue: {method}")
            
        return classes[method](X=target, Y=source, **kwargs)

    def callback(self, iteration, error, X, Y):
        """Fonction appelée à chaque itération par pycpd.register()"""
        print(f"Itération {iteration}, Erreur: {error:.4f}")
        # On peut stocker l'historique ici
        self.history.append(Y.copy())

    def run(self, source_points: np.ndarray, target_points: np.ndarray):
        reg = self._setup_registration(source_points, target_points)
        
        start_time = time.time()
        
        TY, params = reg.register(callback=self.callback)
        
        total_time = time.time() - start_time
        print(f"Enregistrement terminé en {total_time:.2f} secondes.")
        
        return TY