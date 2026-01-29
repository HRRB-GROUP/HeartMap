import numpy as np
import pycpd
import time

class CPDRegistrationWorker:
    def __init__(self, pars):
        self.pars = pars

    def run(self,
            source_points :np.ndarray,
            target_points :np.ndarray
        ):
        
        pars = self.pars['pars']
        method = pars['method']
        n_steps = pars['n_steps']
        kwargs = pars['kwargs']

        print(self.pars)

        if method == 'rigid':
            reg = pycpd.RigidRegistration(X=target_points, Y=source_points, **kwargs)
        elif method == 'affine':
            reg = pycpd.AffineRegistration(X=target_points, Y=source_points, **kwargs)
        elif method == 'deformable':
            if 'source_id' in kwargs and 'target_id' in kwargs:
                print('Constrained deformable registration setup')
                reg = pycpd.ConstrainedDeformableRegistration(X=target_points, Y=source_points, **kwargs)
            else:
                print('deformable registration setup')
                reg = pycpd.DeformableRegistration(X=target_points, Y=source_points, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        iteration_times = []   
        for i in range(1,n_steps+1):
            start_time = time.time()
            reg.iterate()
            elapsed_time = time.time() - start_time
            iteration_times.append(elapsed_time)
            print(f"CPD iteration {i} took {elapsed_time:.4f} seconds")
            #points_history.append(reg.TY.copy())
            print({'state': "running",
                       'source_points': reg.TY,
                       'iter': i})
        
        if iteration_times:
            avg_time = sum(iteration_times) / len(iteration_times)
            print("-" * 30)
            print(f"Average time per iteration: {avg_time:.4f} seconds")
            print("-" * 30)
        return reg.TY