#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import re
import subprocess
import numpy
from numpy import newaxis
from scipy.ndimage import map_coordinates
import nibabel
import cv2

# IO import
from PIL import Image
from slideio import PIL2array


def affine_registration(slices, prefix="a", output_directory=None,
                        thr=20, verbose=1):
    """ Slice to slice affine registration using FSL.

    <process>
        <return name="affine_slices" type="List_File" desc="The registered
            nifti compressed slices."/>
        <input name="slices" type="List_File" desc="The slices to register."/>
        <input name="prefix" type="Str" desc="The registration results
            prefix."/>
        <input name="output_directory" type="Directory" desc="The destination
            folder."/>
        <input name="thr" type="Float" desc="A threshold used to detect
            outliers based on the distance between center of masses."/>
        <input name="verbose" type="Int" desc="If greater than zero display
            debuging information, if greater than one display images."/>
    </process>
    """
    # Compute the center of mass of the first image
    array = nibabel.load(slices[0]).get_data()
    indices = numpy.asarray(numpy.where(array != 0))
    ref_center_of_mass = numpy.mean(indices, axis=1)

    # Go through each slice
    nb_slices = len(slices)
    outliers = []
    for index in range(1, nb_slices):

        # Generate FSL command
        fname = os.path.basename(slices[index]).split(".")[0]
        outfile = os.path.join(output_directory, prefix + fname)
        cmd = ["flirt", "-in", slices[index], "-ref", slices[index - 1],
               "-out", outfile + ".nii.gz", "-omat", outfile + ".txt",
               "-cost", "corratio", "-2D"]
        if verbose > 0:
            print "-" * 10
            print " ".join(cmd)
            print "-" * 10

        # Execute the FSL command
        returncode = subprocess.check_call(cmd)

        # Compute center of mass
        nifti_image = nibabel.load(outfile + ".nii.gz")
        indices = numpy.asarray(numpy.where(nifti_image.get_data() != 0))
        center_of_mass = numpy.mean(indices, axis=1)
        distance = numpy.linalg.norm(ref_center_of_mass - center_of_mass)
        if verbose > 0:
            print "-" * 10
            print "Distance: ", distance
            print "-" * 10

        # Save outputs and update output
        if distance < thr:
            ref_center_of_mass = center_of_mass
            image = Image.fromarray(numpy.cast[numpy.uint8](
                nifti_image.get_data()))
            image.save(outfile + ".tiff")
            slices[index] = outfile + ".nii.gz"
        else:
            os.remove(outfile + ".nii.gz")
            outliers.append(slices[index])
            slices[index] = slices[index - 1]

    if verbose > 0:
        print "-" * 10
        print "Outliers: ", "\n".join(outliers)
        print "-" * 10

    return slices


def nl_registration(slices, prefix="nl", output_directory=None, verbose=1):
    """ Slice to slice nl registration using OpenCv.

    Based on a dense optical flow computation using the Gunnar Farneback's
    algorithm.

    <process>
        <return name="affine_slices" type="List_File" desc="The registered
            nifti compressed slices."/>
        <input name="slices" type="List_File" desc="The slices to register."/>
        <input name="prefix" type="Str" desc="The registration results
            prefix."/>
        <input name="output_directory" type="Directory" desc="The destination
            folder."/>
        <input name="verbose" type="Int" desc="If greater than zero display
            debuging information, if greater than one display images."/>
    </process>
    """
    # Go through each slice
    nb_slices = len(slices)
    for index in range(1, nb_slices):

        # Load slices to register
        prev_image = nibabel.load(slices[index - 1])
        affine = prev_image.get_affine()
        prev_array = prev_image.get_data()
        next_array = nibabel.load(slices[index]).get_data()

        # Compute the output file name
        fname = os.path.basename(slices[index]).split(".")[0]
        outfile = os.path.join(output_directory, prefix + fname)

        # Compute the optical flow
        if verbose > 0:
            print "-" * 10
            print "Dense optical flow computation."
            print "Prev: ", slices[index - 1]
            print "Next: ", slices[index ]
            print "-" * 10
        flow = cv2.calcOpticalFlowFarneback(
            prev_array, next_array, 0.5, 3, 80, 100, 7, 1.5,
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        hsv = numpy.zeros(flow.shape[:-1] + (3, ), dtype=numpy.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[..., 0] = ang * 180 / numpy.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Resample the slice
        ra = apply_cv2_warp(flow, prev_array, next_array)
        resample_image = Image.fromarray(numpy.cast[numpy.uint8](ra))

        # Save outputs
        resample_image.save(outfile + ".tiff")
        nifti_image = nibabel.Nifti1Image(PIL2array(resample_image), affine)
        nibabel.save(nifti_image, outfile + ".nii.gz")
        nifti_image = nibabel.Nifti1Image(flow, affine)
        nibabel.save(nifti_image, outfile + "-warp.nii.gz")
        cv2.imwrite(outfile + "-grid.tiff", rgb)

        # Update output
        slices[index] = outfile + ".nii.gz"

    return slices


def flirt2aff(mat, in_img, ref_img):
    """ Transform from `in_img` voxels to `ref_img` voxels given `mat`

    Parameters
    ----------
    mat : (4,4) array
        contents (as array) of output ``-omat`` transformation file from flirt
    in_img : img
        image passed (as filename) to flirt as ``-in`` image
    ref_img : img
        image passed (as filename) to flirt as ``-ref`` image

    Returns
    -------
    aff : (4,4) array
        Transform from voxel coordinates in ``in_img`` to voxel coordinates in
        ``ref_img``
    """
    in_hdr = in_img.get_header()
    ref_hdr = ref_img.get_header()
    # get_zooms gets the positive voxel sizes as returned in the header
    inspace = numpy.diag(in_hdr.get_zooms() + (1, 1))
    refspace = numpy.diag(ref_hdr.get_zooms() + (1, 1))
    if numpy.linalg.det(in_img.get_affine()) >= 0:
        inspace = numpy.dot(inspace, _x_flipper(in_hdr.get_data_shape()[0]))
    if numpy.linalg.det(ref_img.get_affine()) >= 0:
        refspace = numpy.dot(refspace, _x_flipper(ref_hdr.get_data_shape()[0]))
    # Return voxel to voxel mapping
    return numpy.dot(numpy.linalg.inv(refspace), numpy.dot(mat, inspace))


def _x_flipper(N_i):
    flipr = numpy.diag([-1, 1, 1, 1])
    flipr[0, 3] = N_i - 1
    return flipr


def flirt2aff_files(matfile, in_fname, ref_fname):
    """ Map from `in_fname` image voxels to `ref_fname` voxels given `matfile`
    See :func:`flirt2aff` docstring for details.

    Parameters
    ------------
    matfile : str
        filename of output ``-omat`` transformation file from flirt
    in_fname : str
        filename for image passed to flirt as ``-in`` image
    ref_fname : str
        filename for image passed to flirt as ``-ref`` image

    Returns
    -------
    aff : (4,4) array
        Transform from voxel coordinates in image for ``in_fname`` to voxel
        coordinates in image for ``ref_fname``
    """
    mat = numpy.loadtxt(matfile)
    in_img = nibabel.load(in_fname)
    ref_img = nibabel.load(ref_fname)
    return flirt2aff(mat, in_img, ref_img)


def combine_deformations(ref_images, affine_trfs, nl_fields, ref_index,
                         prefix="low-", output_directory=None, verbose=1):
    """ Combine an affine transformation with a non-linear field.
    """
    # Create deformation mappings
    affine_map = dict((re.findall(r"\d+", os.path.basename(f))[0], f)
                      for f in affine_trfs)
    nl_map = dict((re.findall(r"\d+", os.path.basename(f))[0], f)
                      for f in nl_fields)

    # Go through all deformations
    ref = ref_images[ref_index]
    reference = nibabel.load(ref).get_data()
    ref_shape = reference.shape
    ref_affine = nibabel.load(ref).get_affine()

    ref_spacing = nibabel.load(ref).get_header()["pixdim"][1: 3]
    warped_images = []
    for mov in ref_images:

        # Check that this image was not an outlier
        index = re.findall(r"\d+", os.path.basename(mov))[0]
        if index in nl_map:
            affine = affine_map[index]
            nl = nl_map[index]

            if verbose > 0:
                print "-" * 10
                print "Ref:", ref
                print "Moving:", mov
                print "Affine:", affine
                print "Nl:", nl
        else:
            continue

        # Load moving image
        moving_im = nibabel.load(mov)
        moving = moving_im.get_data()
        spacing = moving_im.get_header().get_zooms()

        # Load displacement field
        displacement = nibabel.load(nl).get_data()

        # From moving index to reference index
        res = flirt2aff_files(affine, mov, ref)

        # From reference index to moving index
        ires = numpy.linalg.inv(res)

        # Create the grid indice for the reference
        grid = numpy.zeros(ref_shape + (1, 3))
        grid[..., 0] = numpy.arange(ref_shape[0])[:, newaxis, newaxis]
        grid[..., 1] = numpy.arange(ref_shape[1])[newaxis, :, newaxis]

        # Affine transform from reference index to the moving index
        A = numpy.dot(grid, ires[:3, :3].T) + ires[:3, 3]
        A = A[..., 0, :2]

        # Add the displacements
        A[..., 0] += displacement[..., 1]
        A[..., 1] += displacement[..., 0]

        # Do the interpolation using map coordinates
        di, dj, dl = A.shape
        W = map_coordinates(moving, A.reshape(di * dj , dl).T, order=1).reshape(
            di, dj)
        # W = apply_cv2_warp(displacement, reference, W, order=1)

        # Define output names
        fname = os.path.basename(mov).split(".")[0]
        outfile = os.path.join(output_directory, prefix + fname)

        # Save the warped image
        image = Image.fromarray(numpy.cast[numpy.uint8](W))
        image.save(outfile + "-wim.tiff")
        nifti_image = nibabel.Nifti1Image(W, ref_affine)
        nibabel.save(nifti_image, outfile + "-wim.nii.gz")

        # Save the deformation field
        nifti_image = nibabel.Nifti1Image(A, ref_affine)
        nibabel.save(nifti_image, outfile + "-warp.nii.gz")        

        if verbose > 0:
            print "Wap: ", outfile + "-warp.nii.gz"
            print "-" * 10

    return warped_images


def apply_cv2_warp(warp, prev, next, order=1):
    """ Apply an open cv warp.
    """
    # Create the grid indice for the reference
    grid = numpy.zeros(prev.shape + (2, ))
    grid[..., 0] = numpy.arange(prev.shape[0])[:, newaxis] + warp[..., 1]
    grid[..., 1] = numpy.arange(prev.shape[1])[newaxis, :] + warp[..., 0]

    # Do the interpolation using map coordinates
    di, dj, dl = grid.shape
    warped = map_coordinates(next, grid.reshape(di * dj, dl).T,
                             order=order).reshape(di, dj)

    return warped


def register_t1_histo(t1_file, histo_file, prefix="f",
                      output_directory=None, verbose=1):
    """ Register the t1 and the histo.
    """
    fname = os.path.basename(histo_file).split(".")[0]
    outfile = os.path.join(output_directory, prefix + fname)
    trffile = os.path.join(output_directory, prefix + fname + ".trf")

    if verbose > 0:
        print "-" * 10
        print "T1 file:", t1_file
        print "Histo file:", histo_file
        print "-" * 10

    cmd = ["flirt", "-in", histo_file, "-ref", t1_file, "-out", outfile,
           "-omat", trffile, "-dof", "12", "-cost", "normmi", "-usesqform",
           "-v"]
    subprocess.check_call(cmd)

