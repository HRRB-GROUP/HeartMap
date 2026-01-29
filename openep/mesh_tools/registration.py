# OpenEP
# Copyright (c) 2021 OpenEP Collaborators
#
# This file is part of OpenEP.
#
# OpenEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenEP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program (LICENSE.txt).  If not, see <http://www.gnu.org/licenses/>

"""
Tools to edit a mesh: Register 2 meshes together

Based on pyvista and displayed in the main 3D viewer (not in the separate mesh editor).
"""

import openep

import numpy as np
import pickle
import time
import pycpd
from typing import Union

from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from PySide6 import QtWidgets, QtCore

from PySide6.QtCore import QProcess, QObject, QProcess, QSocketNotifier, Signal, Slot, QThread

from pyvistaqt.plotting import BackgroundPlotter
from pyvista.plotting._vtk import vtkActor
from pyvista.core import PolyData
import pyvista.plotting._vtk

from OpenEPGUI.view.mesh_tools.base_mesh_tools_ui import BaseMeshToolsWidget, launch_pick_event
from OpenEPGUI.model.mesh_tools import LandmarkRegistrationModel, CPDRegistrationModel


__all__ = ['RegistrationWidget']

class PipeReaderThread(QThread):
    data_received = Signal(dict)
    finished_received = Signal()

    def __init__(self, conn: Connection):
        super().__init__()
        self.conn = conn
        self._running = True

    def run(self):
        while self._running:
            try:
                msg = self.conn.recv() 
            except EOFError:
                break

            if msg.get("state") == "finished":
                self.finished_received.emit()
                break
            else:
                self.data_received.emit(msg)

    def stop(self):
        self._running = False

class CPDRegistrationWorker:
    def __init__(self, pars):
        self.pars = pars

    def run(self,
            conn : Connection,
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
            
        for i in range(1,n_steps+1):
            reg.iterate()
            conn.send({'state': "running",
                       'source_points': reg.TY,
                       'iter': i})

        conn.send({'state': "finished"})  # Signal completion
        conn.close()  # Close the connection when done

class CPDRegistrationTool:

    def __init__(
        self,
        plotter: BackgroundPlotter,
        from_mesh: PolyData,
        from_mesh_actor: vtkActor,
        from_mesh_name: str,
        to_mesh: PolyData,
        to_mesh_actor: vtkActor,
        to_mesh_name: str,
        cpd_pars: dict,
        ):
        
        self.cpd_pars = cpd_pars # cpd_pars['pars'] auto updates from UI
        self.stop_cb : callable = None
        self.points_history : list = [from_mesh.points.copy()]
        print("POINTS_HISTORY INIT: \n", self.points_history)
        self.last_call_time = None

        self.plotter = plotter
        self.from_mesh = from_mesh
        self.from_mesh_actor = from_mesh_actor
        self.from_mesh_name = from_mesh_name
        self.to_mesh = to_mesh
        self.to_mesh_actor = to_mesh_actor
        self.to_mesh_name = to_mesh_name

        self.from_point_ids = []
        self.to_point_ids = []
        self.from_point_id = None
        self.to_point_id = None

        # self.message  = f"Coherent Point Drift\nSource mesh {self.from_mesh_name}\n"
        # self.message += f"Target mesh {self.to_mesh_name}\n"
        
        self.message  = f"Right-click to select a point on mesh {self.from_mesh_name}\n"
        self.message += f"Shift + right-click to to select a point on mesh {self.to_mesh_name}\n"
        self.message += f"Press 'space' to confirm selected points\n"


        self._original_pickable_actors = self.plotter.pickable_actors.copy()
        self.plotter.pickable_actors = [self.from_mesh_actor, self.to_mesh_actor]
        self._picking_right_clicking_observer = None
        self._space_bar_join_points_observer = None

        self._working: bool = False  # prevent errors from multiple clicks

    @property
    def from_points(self):
        return self.from_mesh.points[self.from_point_ids]

    @property
    def to_points(self):
        return self.to_mesh.points[self.to_point_ids]

    @property
    def n_from_points(self):
        return len(self.from_point_ids)

    @property
    def n_to_points(self):
        return len(self.to_point_ids)
    
    def start_task(self, stop_cb : callable):
        """Start the registration task in a separate process."""
        
        print(f'Selected points ids from={self.from_point_ids} to={self.to_point_ids}')
        if len(self.from_point_ids) > 0:
            self.cpd_pars['pars']['kwargs'].update({'e_alpha': 1e-8, 'source_id': np.array(self.from_point_ids), 'target_id': np.array(self.to_point_ids)})
        
        self.stop_cb = stop_cb
        self.points_history = [self.from_mesh.points.copy()]
        print("POINTS_HISTORY START: \n", self.points_history)

        self.plotter.disable_picking()
        if not self._working:
            # Create a Pipe for communication
            self.parent_conn, self.child_conn = Pipe()

            # Set up the worker process
            self.worker = CPDRegistrationWorker(self.cpd_pars)
            #self.process : Process = None  # Will hold the Process instance

            ## Create a QSocketNotifier to watch the pipe
            #self.notifier = QSocketNotifier(self.parent_conn.fileno(), QSocketNotifier.Read)
            #self.notifier.activated.connect(self.handle_output)

            # Start the process
            self.process = Process(target=self.worker.run,
                                   args=(self.child_conn,
                                         np.array(self.from_mesh.points),
                                         np.array(self.to_mesh.points)
                                         )
                                   )
            # QThread that blocks on recv()
            self.pipe_thread = PipeReaderThread(self.parent_conn)
            self.pipe_thread.data_received.connect(self.handle_output)
            self.pipe_thread.finished_received.connect(self.task_finished)

            self.process.start()
            self.pipe_thread.start()
            self.last_call_time = time.time()
            self._working = True
            # self.progress_bar.setValue(0)
            print("Started registration task.")

    def stop_task(self):
        """Stop the registration task."""
        if self._working and self.process.is_alive():
            self.process.terminate()
            self._working = False
            print("Registration task terminated.")
            self.stop_cb()

    def decimate_mesh(self):
        print('Decimating meshes...')
        print(f"Before decimation: {self.from_mesh.points.shape}, {self.from_mesh.faces.shape}")

        # Get the name of the active scalar array used for coloring
        scalar_name = self.from_mesh.active_scalars_name
        print(f"Active scalar name: {scalar_name}")

        # Decimate the meshes while preserving point data (scalars)
        tmp_from_mesh = self.from_mesh.copy(deep=True)
        tmp_to_mesh = self.to_mesh.copy(deep=True)
        self.from_mesh.decimate(0.5, inplace=True)
        self.to_mesh.decimate(0.5, inplace=True)
        # self.from_mesh.points /= np.abs(self.from_mesh.points).max()
        # self.to_mesh.points /= np.abs(self.to_mesh.points).max()
        print(f"After decimation from_mesh: {self.from_mesh.points.shape}, {self.from_mesh.faces.shape}")
        print(f"After decimation to mesh: {self.to_mesh.points.shape}, {self.to_mesh.faces.shape}")
        
        self.from_mesh = self.from_mesh.interpolate(tmp_from_mesh)
        self.to_mesh = self.to_mesh.interpolate(tmp_to_mesh)

        # # Remove old actors from the plotter
        self.plotter.remove_actor(self.from_mesh_actor)
        self.plotter.remove_actor(self.to_mesh_actor)

        self.from_mesh.set_active_scalars(scalar_name)
        self.to_mesh.set_active_scalars(scalar_name)

        # Re-add the decimated meshes with scalar coloring
        self.from_mesh_actor = self.plotter.add_mesh(
            self.from_mesh, scalars=scalar_name, name='0'
        )
        self.to_mesh_actor = self.plotter.add_mesh(
            self.to_mesh, scalars=scalar_name, name='1'
        )

        # Render the updated plot
        self.plotter.render()
    @Slot(dict)
    def handle_output(self, rx: dict):
        state = rx['state']
        print("STATE : ",state)
        print('RX : ',rx)

        current_time = time.time()
        elapsed_time = current_time - self.last_call_time
        self.last_call_time = current_time

        source_points = rx['source_points']
        iter = rx['iter']

        self.from_mesh.points = source_points
        self.points_history.append(source_points.copy())
        print("POINTS_HISTORY HANDLE_OUTPUT: \n", len(self.points_history))

        self.plotter.remove_actor('_registration_tool_message')
        self.plotter.add_text(
            f'Last iter {len(self.points_history)-1}, took {elapsed_time:.2f} s',
            font_size=12,
            name="_registration_tool_message",
        )

#    @Slot()
#   def handle_output(self, rx: dict):
#        """Handle output from the worker process via the pipe."""
#        while self.parent_conn.poll():
#            rx = self.parent_conn.recv()
#           state = rx['state']
#
#            # Calculate the elapsed time since the last call
#            current_time = time.time()
#            elapsed_time = current_time - self.last_call_time
#            self.last_call_time = current_time  # Update the last call time
#
#            if state == "finished":
#                self.task_finished()
#                
#                #TODO: refactor this        
#                self.plotter.remove_actor('_registration_tool_message')
#                self.plotter.add_text(f'Finished', font_size=12, name="_registration_tool_message")
#                # self.plotter.add_text(f'Finished. Last iter {len(self.points_history)-1}, took {elapsed_time:.2f} s', font_size=12, name="_registration_tool_message")
#            else:             
#                source_points = rx['source_points']
#                iter = rx['iter']
#
#                self.from_mesh.points = source_points
#                # self.from_mesh.points = np.random.rand(*source_points.shape)
#                self.points_history.append(source_points.copy())
#                print("POINTS_HISTORY HANDLE_OUTPUT: \n", self.points_history)
#                
#                self.plotter.remove_actor('_registration_tool_message')
#                self.plotter.add_text(f'Last iter {len(self.points_history)-1}, took {elapsed_time:.2f} s', font_size=12, name="_registration_tool_message")

    @Slot()
    def task_finished(self):
        self._working = False

        if self.pipe_thread.isRunning():
            self.pipe_thread.stop()
            self.pipe_thread.wait()

        if self.process.is_alive():
            self.process.join()

        self.plotter.remove_actor('_registration_tool_message')
        self.plotter.add_text('Finished', font_size=12, name="_registration_tool_message")

        print("Registration task completed.")
        self.stop_cb()

#    @Slot()
#    def task_finished(self):
#        """Handle the task completion."""
#        # self.progress_bar.setValue(100)
#        self._working = False
#        print("Registration task completed!")
#        self.notifier.setEnabled(False)  # Disable the notifier once the task is done
#        self.stop_cb()

    def close(self):
        """Disable the tool and reset the plotter."""
        if self._working:
            self.stop_task()
        self.plotter.clear_text('_registration_tool_message')
        self.plotter.enable_picking()

    def start(self):
        """Setup CPD registration, load settings, etc., prepare to start reg."""

        # Turn off current picker
        self.plotter.disable_picking()        

        self.plotter.add_text(
            self.message, font_size=12, name='_registration_tool_message',
        )
        
    def update_ui(self):
        """Update the text in the plotter based on the text message widget."""
        
        print("Update UI tool")
        print(
            "POINTS_HISTORY UPDATE_UI: \n", self.points_history,
            "view_iter =", self.cpd_pars['pars']['view_iter'],
            "history_len =", len(self.points_history)
        )

        txt = self.cpd_pars['pars']['view_iter']
        print(txt)
        self.plotter.remove_actor('_registration_tool_message')
        self.plotter.add_text(f'Iteration {txt} of {len(self.points_history)-1}', font_size=12, name="_registration_tool_message")
        self.from_mesh.points = self.points_history[self.cpd_pars['pars']['view_iter']]
        
    def close(self):
        """Disable the tool and return the plotter to its original state."""

        self.plotter.remove_actor('_registration_tool_from_point')
        self.plotter.remove_actor('_registration_tool_to_point')
        self.plotter.remove_actor('_registration_tool_points')
        self.plotter.remove_actor('_registration_tool_arrows')
        self.plotter.remove_actor('_registration_tool_message')

        self.plotter.pickable_actors = self._original_pickable_actors
        self._picking_right_clicking_observer = None
        self._space_bar_join_points_observer = None

    def _pick_from_point(self, interactor):
        """Select a point on the from_mesh at the picked loaction."""

        picked_position, picked_actor = launch_pick_event(interactor)

        # Ignore if the point was in the window or the other mesh
        if not picked_actor is self.from_mesh_actor:
            return

        self.from_point_id = np.argmin(
            openep.case.calculate_distance(
                picked_position,
                self.from_mesh.points,
            ),
            axis=1,
        ).item()

        self._add_current_point_mesh(
            point=self.from_mesh.points[self.from_point_id],
            name='_registration_tool_from_point',
        )

    def _pick_to_point(self, interactor):
        """Select a point on the to_mesh at the picked loaction."""

        picked_position, picked_actor = launch_pick_event(interactor)

        # Ignore if the point was in the window or the other mesh
        if not picked_actor is self.to_mesh_actor:
            return

        self.to_point_id = np.argmin(
            openep.case.calculate_distance(
                picked_position,
                self.to_mesh.points,
            ),
            axis=1,
        ).item()

        self._add_current_point_mesh(
            point=self.to_mesh.points[self.to_point_id],
            name='_registration_tool_to_point',
        )

    def _add_current_point_mesh(
        self,
        point: np.ndarray,
        name: str,
    ):
        """Create a mesh and add it to the plotter.

        Args:
            point (np.ndarray): Coordinate at which to add a point
            name (str): Name of the actor
        """

        point = PolyData(point)
        self.plotter.add_mesh(
            point,
            render=True,
            render_points_as_spheres=True,
            color='black',
            name=name,
            reset_camera=False,
            point_size=15,
        )

    def _add_points(self):
        """Add currently-selected points to lists for registration."""

        if self.from_point_id is None or self.to_point_id is None:
            return

        self.from_point_ids.append(self.from_point_id)
        self.to_point_ids.append(self.to_point_id)

        self.from_point_id = None
        self.to_point_id = None

        from_points = self.from_mesh.points[self.from_point_ids]
        to_points = self.to_mesh.points[self.to_point_ids]
        points = np.vstack([from_points, to_points])

        self.plotter.remove_actor('_registration_tool_from_point')
        self.plotter.remove_actor('_registration_tool_to_point')
        self.plotter.remove_actor('_registration_tool_points')
        self.plotter.add_mesh(
            points,
            render=True,
            render_points_as_spheres=True,
            color='blue',
            name='_registration_tool_points',
            reset_camera=False,
            point_size=12,
        )

        # also draw arrows between pairs of points
        self._add_arrows()

    def _add_arrows(self):
        """Add arrows from points on from_mesh to points on to_mesh"""

        from_points = np.array(self.from_mesh.points[self.from_point_ids])
        to_points = np.array(self.to_mesh.points[self.to_point_ids])
        directions = to_points - from_points

        arrow = pyvista._vtk.vtkArrowSource()
        arrow.SetShaftRadius(0.005)
        arrow.SetTipLength(0.05)
        arrow.SetTipRadius(0.01)
        arrow.SetShaftResolution(12)
        arrow.SetTipResolution(12)
        arrow.Update()

        pdata = pyvista.vector_poly_data(from_points, directions)
        glyph3D = pyvista._vtk.vtkGlyph3D()
        glyph3D.SetSourceData(arrow.GetOutput())
        glyph3D.SetInputData(pdata)
        glyph3D.SetVectorModeToUseVector()
        glyph3D.Update()

        arrows = pyvista.utilities.wrap(glyph3D.GetOutput())

        # Don't show the selected pairs of points - draw an arrow only.
        # For some reason, if we also show the points as spheres, the GUI
        # occassionally crashes
        self.plotter.remove_actor('_registration_tool_arrows')
        self.plotter.add_mesh(
            arrows,
            name='_registration_tool_arrows',
            color='black',
        )

class CPDRegistrationWidget(QtWidgets.QWidget):
    """Widget for selecting CPD registration options and controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pars = {'pars':{}}
        self.init_ui()
        self.update_pars_from_ui()

    def init_ui(self):
        # Set up layout
        cpd_registration_layout = QtWidgets.QVBoxLayout()

        # Create radio buttons for CPD registration options
        self.cpd_registration_rigid = QtWidgets.QRadioButton("Rigid")
        self.cpd_registration_rigid.setChecked(True)
        self.cpd_registration_rigid.setToolTip('Translation, rotation, scaling only.')
        
        self.cpd_registration_affine = QtWidgets.QRadioButton("Affine")
        self.cpd_registration_affine.setChecked(False)
        self.cpd_registration_affine.setToolTip('Translation, rotation, and non-isotropic scaling.')
        
        self.cpd_registration_deformable = QtWidgets.QRadioButton("Deformable")
        self.cpd_registration_deformable.setChecked(False)
        self.cpd_registration_deformable.setToolTip('All points move individually.')

        # Group the radio buttons to ensure only one can be selected at a time
        cpd_registration_group = QtWidgets.QButtonGroup(self)
        cpd_registration_group.setExclusive(True)
        cpd_registration_group.addButton(self.cpd_registration_rigid)
        cpd_registration_group.addButton(self.cpd_registration_affine)
        cpd_registration_group.addButton(self.cpd_registration_deformable)

        # Add the history file label and line edit on the same line
        history_file_layout = QtWidgets.QHBoxLayout()
        history_file_label = QtWidgets.QLabel("History file:")
        self.history_file_line_edit = QtWidgets.QLineEdit('cpd_history.pkl')
        self.history_file_line_edit.setToolTip('Path to the history file')
        history_file_layout.addWidget(history_file_label)
        history_file_layout.addWidget(self.history_file_line_edit)
        cpd_registration_layout.addLayout(history_file_layout)

        # Add the save and load buttons on the same line
        save_load_layout = QtWidgets.QHBoxLayout()

        # Create the save button
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setToolTip('Save CPD registration history')
        save_load_layout.addWidget(self.save_button)

        # Create the load button
        self.load_button = QtWidgets.QPushButton("Load")
        self.load_button.setToolTip('Load CPD registration history')
        save_load_layout.addWidget(self.load_button)

        # Add the save and load buttons layout to the main layout
        cpd_registration_layout.addLayout(save_load_layout)
        
        # Add the iteration slider
        self.iteration_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iteration_slider.setMinimum(0)
        self.iteration_slider.setMaximum(100)
        self.iteration_slider.setValue(0)
        self.iteration_slider.setTickInterval(1)
        self.iteration_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.iteration_slider.setToolTip('View iteration')
        cpd_registration_layout.addWidget(self.iteration_slider)
        self.iteration_slider.valueChanged.connect(self.update_pars_from_ui)

        # Add the n_steps label and line edit on the same line
        n_steps_layout = QtWidgets.QHBoxLayout()
        n_steps_label = QtWidgets.QLabel("n_steps:")
        self.n_steps_line_edit = QtWidgets.QLineEdit('50')
        self.n_steps_line_edit.setToolTip('Number of iterations.')
        n_steps_layout.addWidget(n_steps_label)
        n_steps_layout.addWidget(self.n_steps_line_edit)
        cpd_registration_layout.addLayout(n_steps_layout)

        # Add the w label and line edit on the same line
        w_layout = QtWidgets.QHBoxLayout()
        w_label = QtWidgets.QLabel("w:")
        self.w_line_edit = QtWidgets.QLineEdit('0')
        self.w_line_edit.setToolTip('Weight of the uniform distribution (amount of noise in the point clouds). 0..1 Default: 0')
        w_layout.addWidget(w_label)
        w_layout.addWidget(self.w_line_edit)
        cpd_registration_layout.addLayout(w_layout)

        # Add widgets to layout
        cpd_registration_layout.addWidget(self.cpd_registration_rigid)
        cpd_registration_layout.addWidget(self.cpd_registration_affine)
        cpd_registration_layout.addWidget(self.cpd_registration_deformable)
        
        # Create the new widget with labels and line edits
        self.deformable_options_widget = QtWidgets.QWidget()
        deformable_options_layout = QtWidgets.QVBoxLayout()

        # Alpha label and line edit on the same line
        alpha_layout = QtWidgets.QHBoxLayout()
        alpha_label = QtWidgets.QLabel("Alpha:")
        self.alpha_line_edit = QtWidgets.QLineEdit('0.002')
        self.alpha_line_edit.setToolTip('A higher value makes the deformation more rigid, a lower value makes the deformation more flexible. Default: 2')
        alpha_layout.addWidget(alpha_label)
        alpha_layout.addWidget(self.alpha_line_edit)
        deformable_options_layout.addLayout(alpha_layout)

        # Beta label and line edit on the same line
        beta_layout = QtWidgets.QHBoxLayout()
        beta_label = QtWidgets.QLabel("Beta:")
        self.beta_line_edit = QtWidgets.QLineEdit('2')
        self.beta_line_edit.setToolTip('Width of the Gaussian kernel used to regularize the deformation. Default: 2')
        beta_layout.addWidget(beta_label)
        beta_layout.addWidget(self.beta_line_edit)
        deformable_options_layout.addLayout(beta_layout)

        # num_eign label and line edit on the same line
        num_eig_layout = QtWidgets.QHBoxLayout()
        num_eign_label = QtWidgets.QLabel("num_eig:")
        self.num_eig_line_edit = QtWidgets.QLineEdit('100')
        self.num_eig_line_edit.setToolTip('Number of eigenvectors for the low-rank approximation. Default: 100')
        num_eig_layout.addWidget(num_eign_label)
        num_eig_layout.addWidget(self.num_eig_line_edit)
        deformable_options_layout.addLayout(num_eig_layout)
        
        self.low_rank_checkbox = QtWidgets.QCheckBox("low_rank")
        self.low_rank_checkbox.setToolTip('Use low-rank approximation for the kernel matrix. Default: false')
        deformable_options_layout.addWidget(self.low_rank_checkbox)

        self.deformable_options_widget.setLayout(deformable_options_layout)
        self.deformable_options_widget.setVisible(False)  # Initially hidden

        cpd_registration_layout.addWidget(self.deformable_options_widget)

        # Set the layout for this widget
        self.setLayout(cpd_registration_layout)

        # Connect the toggled signal to the callback function
        self.cpd_registration_deformable.toggled.connect(self.on_deformable_toggled)

    def update_pars_from_ui(self):
        """Update the CPD parameters from the UI."""
        
        # print(self.pars)
        self.pars['pars'] = {
            'view_iter': self.iteration_slider.value(),
            'method' : 'rigid' if self.cpd_registration_rigid.isChecked() else 'affine' if self.cpd_registration_affine.isChecked() else 'deformable',
            'n_steps': int(self.n_steps_line_edit.text()),
            'kwargs': { 'w': float(self.w_line_edit.text()) }
        }
        
        self.iteration_slider.setMaximum(self.pars['pars']['n_steps'])
        
        if self.cpd_registration_deformable.isChecked():
            self.pars['pars']['kwargs'].update({
                'alpha': float(self.alpha_line_edit.text()),
                'beta': float(self.beta_line_edit.text()),
                'num_eig': int(self.num_eig_line_edit.text()),
                'low_rank': self.low_rank_checkbox.isChecked()
            })
        
    def on_deformable_toggled(self, checked):
        self.deformable_options_widget.setVisible(checked)


class RegistrationWidget(BaseMeshToolsWidget):
    """Layout for registering two surface meshes."""

    def __init__(self, title: str, parent_plotter: BackgroundPlotter):

        super().__init__(title, parent_plotter)

        self.model = LandmarkRegistrationModel()
        self._tool: Union[LandmarkRegistrationTool, CPDRegistrationTool] = None

        self._systems_widget: QtWidgets.QWidget = None
        self._from_system_widget: QtWidgets.QComboBox = None
        self._to_system_widget: QtWidgets.QComboBox = None

        self._register_info_widget: QtWidgets.QWidget = None
        self.select_method_combobox: QtWidgets.QComboBox = None
        self._landmark_registration_group: QtWidgets.QButtonGroup = None

        self.stop_registration_button: QtWidgets.QPushButton = None
        self.start_registration_button: QtWidgets.QPushButton = None
        self._start_registration_button_container: QtWidgets.QWidget = None
        self.cancel_button: QtWidgets.QPushButton = None
        self._confirm_registration_button: QtWidgets.QPushButton = None

        self._create_systems_layout()  # comboboxes for selecting which systems to register
        self._create_method_layout()  # combobox for selecting which tool to use
        self._create_register_button()
        self.add_to_layout()

    @property
    def method(self):
        return self.select_method_combobox.currentText()

    @property
    def from_system_index(self):
        return self._from_system_widget.currentText()

    @property
    def from_system_name(self):
        return self._from_system_widget.currentText()

    @property
    def to_system_index(self):
        return self._to_system_widget.currentText()

    @property
    def to_system_name(self):
        return self._to_system_widget.currentText()

    def _create_systems_layout(self):
        """Create a wdiget for selecting which systems to register"""

        systems = QtWidgets.QWidget()
        systems_layout = QtWidgets.QFormLayout()

        from_system = QtWidgets.QComboBox()
        to_system = QtWidgets.QComboBox()

        systems_layout.addRow("Register", from_system)
        systems_layout.addRow("with", to_system)
        systems.setLayout(systems_layout)

        self._systems_widget = systems
        self._from_system_widget = from_system
        self._to_system_widget = to_system

    def save_cpd_history(self):
        """Save the CPD history to a file."""
        history_file = self.cpd_registration_control_view.history_file_line_edit.text()
        with open(history_file, 'wb') as f:
            pickle.dump(self._tool.points_history, f)
        print("Saved CPD history to file.")
            
    def load_cpd_history(self):
        """Load the CPD history from a file."""
        history_file = self.cpd_registration_control_view.history_file_line_edit.text()
        with open(history_file, 'rb') as f:
            self._tool.points_history = pickle.load(f)
            print('history len', len(self._tool.points_history))
            self.cpd_registration_control_view.iteration_slider.setMaximum(len(self._tool.points_history)-1)
            self.cpd_registration_control_view.n_steps_line_edit.setText(str(len(self._tool.points_history)-1))
            self.cpd_registration_control_view.update_pars_from_ui()
            self._tool.update_ui()
        print("Loaded CPD history from file.")

    def _create_method_layout(self):
        """Create a widget to select which type of registration to perform"""

        register_method = QtWidgets.QWidget()
        register_method_layout = QtWidgets.QVBoxLayout()

        # Setup list of available methods
        self.select_method_combobox = QtWidgets.QComboBox()
        self.select_method_combobox.setMinimumWidth(200)
        self.select_method_combobox.setEditable(False)
        self.select_method_combobox.addItem("Landmark registration")
        self.select_method_combobox.addItem("Coherent Point Drift registration")
        self.select_method_combobox.activated.connect(self.change_selected_tool)

        # Layouts for the different methods
        self.tool_layout = QtWidgets.QStackedLayout()

        # Landmark registration
        landmark_registration = QtWidgets.QWidget()
        landmark_registration_layout = QtWidgets.QVBoxLayout()

        landmark_registration_rigid = QtWidgets.QRadioButton("Rigid")
        landmark_registration_rigid.setChecked(False)
        landmark_registration_rigid.setToolTip('Translation and rotation only.')
        landmark_registration_similarity = QtWidgets.QRadioButton("Similarity")
        landmark_registration_similarity.setChecked(True)
        landmark_registration_similarity.setToolTip('Translation, rotation, and isotropic scaling.')
        landmark_registration_affine = QtWidgets.QRadioButton("Affine")
        landmark_registration_affine.setChecked(False)
        landmark_registration_affine.setToolTip('Translation, rotation, and non-isotropic scaling.')

        landmark_registration_group = QtWidgets.QButtonGroup()
        landmark_registration_group.setExclusive(True)
        landmark_registration_group.addButton(landmark_registration_rigid)
        landmark_registration_group.addButton(landmark_registration_similarity)
        landmark_registration_group.addButton(landmark_registration_affine)

        landmark_registration_layout.addWidget(landmark_registration_rigid)
        landmark_registration_layout.addWidget(landmark_registration_similarity)
        landmark_registration_layout.addWidget(landmark_registration_affine)
        landmark_registration.setLayout(landmark_registration_layout)
        self.tool_layout.addWidget(landmark_registration)


        # CPD registration
        self.cpd_registration_control_view = CPDRegistrationWidget() #TODO: separate model, view and controler
        self.cpd_registration_control_view.save_button.clicked.connect(self.save_cpd_history)
        self.cpd_registration_control_view.load_button.clicked.connect(self.load_cpd_history)
        self.tool_layout.addWidget(self.cpd_registration_control_view)

        # Set layout
        register_method_layout.addWidget(self.select_method_combobox, 0)
        register_method_layout.addLayout(self.tool_layout, 0)
        register_method_layout.addStretch(1)
        register_method.setLayout(register_method_layout)

        self._register_info_widget = register_method
        self._landmark_registration_group = landmark_registration_group
        self._landmark_registration_group.buttonReleased.connect(self._update_landmark_registration_method)

    def _create_register_button(self):
        """
        Create button for performing registration.

        Put this in a containing widget so we can move the button to always be on the
        right hand side of the RegistrationWidget.
        """

        # show these only when adding a landmark
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.setVisible(False)
        
        confirm_registration_button = QtWidgets.QPushButton('Register')
        confirm_registration_button.setVisible(False)
        confirm_registration_button.setEnabled(False)
        
        self.stop_registration_button = QtWidgets.QPushButton('Stop')
        self.stop_registration_button.setVisible(False)
        self.stop_registration_button.setEnabled(False)

        self.decimate_button = QtWidgets.QPushButton("Decimate")
        self.decimate_button.hide()

        register_button = QtWidgets.QPushButton("Start registration")
        register_button.setEnabled(False)  # only enable once there are at least two systems loaded into the GUI

        layout = QtWidgets.QHBoxLayout()
        layout.addStretch(1)
        layout.addWidget(self.decimate_button)
        layout.addWidget(cancel_button)
        layout.addWidget(confirm_registration_button)
        layout.addWidget(register_button)
        layout.addWidget(self.stop_registration_button)

        register_button_container = QtWidgets.QWidget()
        register_button_container.setLayout(layout)

        self.start_registration_button = register_button
        self._start_registration_button_container = register_button_container
        self.cancel_button = cancel_button
        self._confirm_registration_button = confirm_registration_button

    def add_to_layout(self):
        """Create the layout for the landmark picker."""

        self.layout.addWidget(self._systems_widget, 0)
        self.layout.addWidget(self._register_info_widget, 0)
        self.layout.addStretch(1)
        self.layout.addWidget(self._start_registration_button_container, 0)

        self._register_info_widget.setVisible(False)

    def add_system(self, system):
        """Add a system to those available for registration."""
        self._from_system_widget.addItem(system.name)
        self._to_system_widget.addItem(system.name)
        if self._from_system_widget.count() > 1:
            self.start_registration_button.setEnabled(True)

    def update_from_system(self, system):
        """Update the model with a new system."""
        self.model.from_system = system

    def update_to_system(self, system):
        """Update the model with a new system."""
        self.model.to_system = system

    def start_registration(self, from_mesh_actor, to_mesh_actor):
        """Interactively define a new landmark on the surface mesh."""

        self.start_registration_button.setVisible(False)
        self._systems_widget.setEnabled(False)
        self._register_info_widget.setVisible(True)
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)
        self._confirm_registration_button.setVisible(True)
        self._confirm_registration_button.setEnabled(False)

        if self.select_method_combobox.currentText() == 'Landmark registration':
            self._tool = LandmarkRegistrationTool(
                plotter=self._parent_plotter,
                from_mesh=self.model.from_mesh,
                from_mesh_name=self.from_system_name,
                to_mesh=self.model.to_mesh,
                from_mesh_actor=from_mesh_actor,
                to_mesh_actor=to_mesh_actor,
                to_mesh_name=self.to_system_name,
            )
        elif self.select_method_combobox.currentText() == 'Coherent Point Drift registration':
            self._tool = CPDRegistrationTool(
                plotter=self._parent_plotter,
                from_mesh=self.model.from_mesh,
                from_mesh_name=self.from_system_name,
                to_mesh=self.model.to_mesh,
                from_mesh_actor=from_mesh_actor,
                to_mesh_actor=to_mesh_actor,
                to_mesh_name=self.to_system_name,
                cpd_pars=self.cpd_registration_control_view.pars,
            )
            self.stop_registration_button.clicked.connect(self._tool.stop_task)
                    
        self.tool_layout.setCurrentIndex(self.select_method_combobox.currentIndex())

    def close_tool(self):
        """Wrapper around self.cancel_add_landmark"""
        self.cancel_registration()

    def cancel_registration(self):
        """Delete the currently-defined landmark to be added."""

        self.cancel_button.setVisible(False)
        self._confirm_registration_button.setVisible(False)
        self._confirm_registration_button.setEnabled(False)
        self._register_info_widget.setVisible(False)
        self.start_registration_button.setVisible(True)
        self._systems_widget.setEnabled(True)

        # Remove callbacks
        self._parent_plotter.iren.remove_observer(self._tool._picking_right_clicking_observer)
        self._parent_plotter.iren.remove_observer(self._tool._space_bar_join_points_observer)

        self._tool.close()
        self._tool = None

    def confirm_registration(self):

        if isinstance(self._tool, LandmarkRegistrationTool):
            transform_matrix = self.model.get_transform_matrix(
                source_points=self._tool.from_points,
                target_points=self._tool.to_points,
            )
        self.cancel_button.setVisible(False)
        self._confirm_registration_button.setVisible(False)
        self._confirm_registration_button.setEnabled(False)
        self._register_info_widget.setVisible(False)
        self.start_registration_button.setVisible(True)
        self._systems_widget.setEnabled(True)

        # Remove callbacks
        self._parent_plotter.disable_picking()
        self._parent_plotter.iren.remove_observer(self._tool._picking_right_clicking_observer)
        self._parent_plotter.iren.remove_observer(self._tool._space_bar_join_points_observer)

        # disable the tool
        self._tool.close()
        if isinstance(self._tool, CPDRegistrationTool):
            self._tool = None
            return
        else:
            self._tool = None
            return transform_matrix

    def change_selected_tool(self):
        """Close the current tool and start the selected one."""

        current_tool = self._tool
        plotter = current_tool.plotter
        from_mesh = current_tool.from_mesh
        from_mesh_actor = current_tool.from_mesh_actor
        from_mesh_name = current_tool.from_mesh_name
        to_mesh = current_tool.to_mesh
        to_mesh_actor = current_tool.to_mesh_actor
        to_mesh_name = current_tool.to_mesh_name
        current_tool.close()
        
        self.tool_layout.setCurrentIndex(self.select_method_combobox.currentIndex())

        method = self.select_method_combobox.currentText()
        if method == 'Landmark registration':
            # self._confirm_button.setVisible(False)
            self._confirm_registration_button.setEnabled(False)
            self._tool = LandmarkRegistrationTool(
                plotter=plotter,
                from_mesh=from_mesh,
                from_mesh_name=from_mesh_name,
                to_mesh=to_mesh,
                from_mesh_actor=from_mesh_actor,
                to_mesh_actor=to_mesh_actor,
                to_mesh_name=to_mesh_name,
                )

            self.decimate_button.hide()

        elif method == 'Coherent Point Drift registration':
            # self._confirm_button.setVisible(True)
            self._confirm_registration_button.setEnabled(True)
            #TODO: CPD tool
            self._tool = CPDRegistrationTool(
                plotter=plotter,
                from_mesh=from_mesh,
                from_mesh_name=from_mesh_name,
                to_mesh=to_mesh,
                from_mesh_actor=from_mesh_actor,
                to_mesh_actor=to_mesh_actor,
                to_mesh_name=to_mesh_name,
                cpd_pars=self.cpd_registration_control_view.pars,
            )
            self.decimate_button.show()
            self.decimate_button.pressed.connect(self._tool.decimate_mesh)
            self.stop_registration_button.clicked.connect(self._tool.stop_task)
            self.cpd_registration_control_view.iteration_slider.valueChanged.connect(self._tool.update_ui)

        self._tool.start()

    def _update_landmark_registration_method(self):
        """Update the method to be used for landmark registration.

        Will be one of:
            - rigid
            - similarity
            - affine
        """

        buttons = self._landmark_registration_group.buttons()

        for button in buttons:
            if button.text == 'Rigid':
                self.tool.rigid = button.isChecked()
            elif button.text == 'Similarity':
                self.tool.similarity = button.isChecked()
            elif button.text == 'Affine':
                self.tool.similarity = button.isChecked()


class LandmarkRegistrationTool:

    def __init__(
            self,
            plotter: BackgroundPlotter,
            from_mesh: PolyData,
            from_mesh_actor: vtkActor,
            from_mesh_name: str,
            to_mesh: PolyData,
            to_mesh_actor: vtkActor,
            to_mesh_name: str,
    ):

        self.plotter = plotter
        self.from_mesh = from_mesh
        self.from_mesh_actor = from_mesh_actor
        self.from_mesh_name = from_mesh_name
        self.to_mesh = to_mesh
        self.to_mesh_actor = to_mesh_actor
        self.to_mesh_name = to_mesh_name

        self.from_point_ids = []
        self.to_point_ids = []
        self.from_point_id = None
        self.to_point_id = None

        self.message = f"Right-click to select a point on mesh {self.from_mesh_name}\n"
        self.message += f"Shift + right-click to to select a point on mesh {self.to_mesh_name}\n"
        self.message += f"Press 'space' to confirm selected points\n"

        self._original_pickable_actors = self.plotter.pickable_actors.copy()
        self.plotter.pickable_actors = [self.from_mesh_actor, self.to_mesh_actor]
        self._picking_right_clicking_observer = None
        self._space_bar_join_points_observer = None

        self._working: bool = False  # prevent errors from multiple clicks

    @property
    def from_points(self):
        return self.from_mesh.points[self.from_point_ids]

    @property
    def to_points(self):
        return self.to_mesh.points[self.to_point_ids]

    @property
    def n_from_points(self):
        return len(self.from_point_ids)

    @property
    def n_to_points(self):
        return len(self.to_point_ids)

    def start(self):
        """Setup the path picking actions"""

        #  Turn off current picker
        self.plotter.disable_picking()

        self.plotter.add_text(
            self.message, font_size=12, name='_registration_tool_message',
        )

    def close(self):
        """Disable the tool and return the plotter to its original state."""

        self.plotter.remove_actor('_registration_tool_from_point')
        self.plotter.remove_actor('_registration_tool_to_point')
        self.plotter.remove_actor('_registration_tool_points')
        self.plotter.remove_actor('_registration_tool_arrows')
        self.plotter.remove_actor('_registration_tool_message')

        self.plotter.pickable_actors = self._original_pickable_actors
        self._picking_right_clicking_observer = None
        self._space_bar_join_points_observer = None

    def _pick_from_point(self, interactor):
        """Select a point on the from_mesh at the picked loaction."""

        picked_position, picked_actor = launch_pick_event(interactor)

        # Ignore if the point was in the window or the other mesh
        if not picked_actor is self.from_mesh_actor:
            return

        self.from_point_id = np.argmin(
            openep.case.calculate_distance(
                picked_position,
                self.from_mesh.points,
            ),
            axis=1,
        ).item()

        self._add_current_point_mesh(
            point=self.from_mesh.points[self.from_point_id],
            name='_registration_tool_from_point',
        )

    def _pick_to_point(self, interactor):
        """Select a point on the to_mesh at the picked loaction."""

        picked_position, picked_actor = launch_pick_event(interactor)

        # Ignore if the point was in the window or the other mesh
        if not picked_actor is self.to_mesh_actor:
            return

        self.to_point_id = np.argmin(
            openep.case.calculate_distance(
                picked_position,
                self.to_mesh.points,
            ),
            axis=1,
        ).item()

        self._add_current_point_mesh(
            point=self.to_mesh.points[self.to_point_id],
            name='_registration_tool_to_point',
        )

    def _add_current_point_mesh(
            self,
            point: np.ndarray,
            name: str,
    ):
        """Create a mesh and add it to the plotter.

        Args:
            point (np.ndarray): Coordinate at which to add a point
            name (str): Name of the actor
        """

        point = PolyData(point)
        self.plotter.add_mesh(
            point,
            render=True,
            render_points_as_spheres=True,
            color='black',
            name=name,
            reset_camera=False,
            point_size=15,
        )

    def _add_points(self):
        """Add currently-selected points to lists for registration."""

        if self.from_point_id is None or self.to_point_id is None:
            return

        self.from_point_ids.append(self.from_point_id)
        self.to_point_ids.append(self.to_point_id)

        self.from_point_id = None
        self.to_point_id = None

        from_points = self.from_mesh.points[self.from_point_ids]
        to_points = self.to_mesh.points[self.to_point_ids]
        points = np.vstack([from_points, to_points])

        self.plotter.remove_actor('_registration_tool_from_point')
        self.plotter.remove_actor('_registration_tool_to_point')
        self.plotter.remove_actor('_registration_tool_points')
        self.plotter.add_mesh(
            points,
            render=True,
            render_points_as_spheres=True,
            color='blue',
            name='_registration_tool_points',
            reset_camera=False,
            point_size=12,
        )

        # also draw arrows between pairs of points
        self._add_arrows()

    def _add_arrows(self):
        """Add arrows from points on from_mesh to points on to_mesh"""

        from_points = np.array(self.from_mesh.points[self.from_point_ids])
        to_points = np.array(self.to_mesh.points[self.to_point_ids])
        directions = to_points - from_points

        arrow = pyvista._vtk.vtkArrowSource()
        arrow.SetShaftRadius(0.005)
        arrow.SetTipLength(0.05)
        arrow.SetTipRadius(0.01)
        arrow.SetShaftResolution(12)
        arrow.SetTipResolution(12)
        arrow.Update()

        pdata = pyvista.vector_poly_data(from_points, directions)
        glyph3D = pyvista._vtk.vtkGlyph3D()
        glyph3D.SetSourceData(arrow.GetOutput())
        glyph3D.SetInputData(pdata)
        glyph3D.SetVectorModeToUseVector()
        glyph3D.Update()

        arrows = pyvista.utilities.wrap(glyph3D.GetOutput())

        # Don't show the selected pairs of points - draw an arrow only.
        # For some reason, if we also show the points as spheres, the GUI
        # occassionally crashes
        self.plotter.remove_actor('_registration_tool_arrows')
        self.plotter.add_mesh(
            arrows,
            name='_registration_tool_arrows',
            color='black',
        )