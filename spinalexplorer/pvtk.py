#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy
import types
import logging

# VTK import
import vtk
from vtk.util.vtkConstants import VTK_UNSIGNED_CHAR

# ITK import
import SimpleITK as sitk


def ren():
    """ Create a renderer.

    Returns
    --------
    ren: vtkRenderer() object

    Examples
    --------
    >>> import plot_vtk
    >>> ren = plot_vtk.ren()
    >>> plot_vtk.add(ren, actor)
    >>> plot_vtk.show(ren)
    """
    return vtk.vtkRenderer()


def add(ren, actor):
    """ Add a specific actor.
    """
    if isinstance(actor, vtk.vtkVolume):
        ren.AddVolume(actor)
    else:
        ren.AddActor(actor)


def rm(ren, actor):
    """ Remove a specific actor.
    """
    ren.RemoveActor(actor)


def clear(ren):
    """ Remove all actors from the renderer.
    """
    ren.RemoveAllViewProps()


def show(ren, title="pvtk", size=(300, 300)):
    """ Show window.

    Parameters
    ----------
    ren : vtkRenderer() object
        as returned from function ren()
    title : string
        a string for the window title bar
    size : (int, int)
        (width,height) of the window
    """
    ren.ResetCameraClippingRange()

    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    window.SetWindowName(title)
    window.SetSize(size)

    style = vtk.vtkInteractorStyleTrackballCamera()
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(window)
    iren.SetInteractorStyle(style)
    iren.Initialize()

    window.Render()
    iren.Start()


def mesh(binary_image):
    """ Create a mesh actor.
    """
    # Load volume
    image = sitk.ReadImage(binary_image)
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    array = sitk.GetArrayFromImage(image)
    importer = numpy2vtk(array, spacing, origin)

    # Create the mesh
    discrete = vtk.vtkDiscreteMarchingCubes() 
    discrete.SetInputConnection(importer.GetOutputPort())

    # Smooth the mesh
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(discrete.GetOutputPort())
    smooth.SetNumberOfIterations(20)

    # Create mapper
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smooth.GetOutputPort())
    normals.FlipNormalsOn()
    mapper = vtk.vtkPolyDataMapper() 
    mapper.SetInputConnection(normals.GetOutputPort())

    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1)

    return actor


def numpy2vtk(array, spacing=[1.0, 1.0, 1.0], origin=[0, 0, 0]):
    """ Convert a numpy array in vtk image.
    """
    importer = vtk.vtkImageImport()
    data = array.astype("uint8")
    data_string = data.tostring()
    dim = array.shape
    
    importer.CopyImportVoidPointer(data_string, len(data_string))
    importer.SetDataScalarType(VTK_UNSIGNED_CHAR)
    importer.SetNumberOfScalarComponents(1)
    
    extent = importer.GetDataExtent()
    importer.SetDataExtent(extent[0], extent[0] + dim[2] - 1,
                           extent[2], extent[2] + dim[1] - 1,
                           extent[4], extent[4] + dim[0] - 1)
    importer.SetWholeExtent(extent[0], extent[0] + dim[2] - 1,
                            extent[2], extent[2] + dim[1] - 1,
                            extent[4], extent[4] + dim[0] - 1)
 
    importer.SetDataSpacing(spacing[0], spacing[1], spacing[2])
    importer.SetDataOrigin(origin[0], origin[1], origin[2])
 
    return importer


if __name__ == "__main__":

    binary_image = (
        "/usr/share/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz")
    ren = ren()
    actor = mesh(binary_image)
    add(ren, actor)
    show(ren)

