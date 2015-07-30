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
import numpy
import json

# IO import
import openslide
from PIL import Image
import nibabel


def get_slices(rootpath, regex, sanity=False, verbose=1):
    """ Get the ordered slices contained in a root folder.

    <process>
        <return name="slices" type="List_File" desc="The ordered slices."/>
        <input name="rootpath" type="Directory" desc="The path
            where the slices are searched recursively."/>
        <input name="regex" type="Str" desc="A regular expression used to
            locate the desired slices."/>
        <input name="verbose" type="Int" desc="If greater than zero display
            debuging information, if greater than one display images."/>
    </process>
    """
    # Load the sanity file
    if sanity:
        sanity_file = os.path.join(rootpath, "sanity.json")
        sanity = json.load(open(sanity_file, "r"))

    # Parse the rootpath
    slices = []
    for root, dirs, files in os.walk(rootpath):
        for fname in files:
            if not sanity and len(re.findall(regex, fname)) > 0:
                slices.append(os.path.join(root, fname))
            elif len(re.findall(regex, fname)) > 0:
                slice_number = fname.split(".")[0]
                if slice_number in sanity and sanity[slice_number] == 1:
                    slices.append(os.path.join(root, fname))

    
    # Order the slices
    slices = sorted(slices)
    if verbose > 0:
        print "-" * 10
        print "Slices:"
        print "\n".join(slices)
        print "-" * 10

    return slices


def downsample_slices(slices, downsample_factor=10, prefix="g",
                      output_directory=None, verbose=1):
    """ Open a slice image with openslide, downsample it, create a grayscale
    image and save the result as '.tiff' and '.nii.gz' images.

    <process>
        <return name="converted_slices" type="List_File" desc="The compressed
            nifti downsample grayscale input slices."/>
        <input name="slices" type="List_File" desc="The slices to downsample."/>
        <input name="downsample_factor" type="Int" desc="Parameter specifying
            the downsample image scale."/>
        <input name="prefix" type="Str" desc="The downsample output image
            prefix."/>
        <input name="output_directory" type="Directory" desc="The destination
            folder."/>
        <input name="verbose" type="Int" desc="If greater than zero display
            debuging information, if greater than one display images."/>
    </process>
    """
    # Check some slices have been passed as an input
    if len(slices) == 0:
        raise ValueError("No slice has been specified.".format(
            output_directory))

    # Check that the outdir is valid
    output_directory = output_directory or os.path.dirname(slices[0])
    if not os.path.isdir(output_directory):
        raise ValueError("'{0}' is not a valid directory.".format(
            output_directory))

    # Open, downsample and convert each slice
    converted_slices = []
    for filename in slices:

        if verbose > 0:
            print "-" * 10
            print "File:", filename

        # Open - downsample - grayscale
        slide = openslide.OpenSlide(filename)
        if verbose > 0:
            print "Nb of levels:", slide.level_count
            print "Level dims:", slide.level_dimensions
            print "Level downsamples:", slide.level_downsamples
            print "Best level:", slide.get_best_level_for_downsample(
                downsample_factor)
            print "Spacing x:", slide.properties[openslide.PROPERTY_NAME_MPP_X]
            print "Spacing y:", slide.properties[openslide.PROPERTY_NAME_MPP_Y]
            print "-" * 10
        downsample_size = tuple(numpy.round(
            numpy.asarray(slide.level_dimensions[0]) / downsample_factor))
        try:
            downsample_image = slide.get_thumbnail(downsample_size).convert("L")
        except:
            print "*** ERROR:", filename
            raise
        spacing = numpy.array([
            float(slide.properties[openslide.PROPERTY_NAME_MPP_X]) * 1e-3,
            float(slide.properties[openslide.PROPERTY_NAME_MPP_X]) * 1e-3])
        spacing *= downsample_factor
        slide.close()

        # Save result
        fname = os.path.basename(filename).split(".")[0]
        converted_file = os.path.join(output_directory, prefix + fname)
        downsample_image.save(converted_file + ".tiff")
        affine = numpy.eye(4)
        affine[:2, :2] = numpy.diag(spacing)
        nifti_image = nibabel.Nifti1Image(
            PIL2array(downsample_image), affine)
        nibabel.save(nifti_image, converted_file + ".nii.gz")

        # Update the output
        converted_slices.append(converted_file + ".nii.gz")

    return converted_slices


def PIL2array(image):
    """ Convert  a pillow image to a numpy array.
    """
    return numpy.array(image.getdata(), numpy.single).reshape(
        image.size[1], image.size[0])


def stack_slices(slices, slice_offset, slice_orders=None, prefix="v",
                 output_directory=None, verbose=1):
    """ Open a slice image with openslide, downsample it, create a grayscale
    image and save the result as '.tiff' and '.nii.gz' images.

    <process>
        <return name="volume_file" type="File" desc="The compressed
            nifti volume containing the input slices."/>
        <input name="slices" type="List_File" desc="The slices to stack."/>
        <input name="slice_offset" type="Int" desc="Parameter specifying
            the slice to slice distance."/>
        <input name="slice_orders" type="List" content="Str" desc="The
            slice orders in the volume."/>
        <input name="prefix" type="Str" desc="The downsample output image
            prefix."/>
        <input name="output_directory" type="Directory" desc="The destination
            folder."/>
        <input name="verbose" type="Int" desc="If greater than zero display
            debuging information, if greater than one display images."/>
    </process>
    """
    # Check some slices have been passed as an input
    if len(slices) == 0:
        raise ValueError("No slice has been specified.".format(
            output_directory))

    # Check that the outdir is valid
    output_directory = output_directory or os.path.dirname(slices[0])
    if not os.path.isdir(output_directory):
        raise ValueError("'{0}' is not a valid directory.".format(
            output_directory))

    # Open, downsample and convert each slice
    slice_0 = nibabel.load(slices[0])
    spacing = slice_0.get_header()["pixdim"][1: 3].tolist() + [slice_offset, 1]
    print spacing
    affine = numpy.diag(spacing)

    if slice_orders is not None:
        volume_array = numpy.zeros(slice_0.get_shape() + (len(slice_orders), ),
                                   dtype=numpy.uint8)
        for filename in slices:
            index = slice_orders.index(
                re.findall(r"\d+", os.path.basename(filename))[0])
            volume_array[..., index] = nibabel.load(filename).get_data()
    else:
        volume_array = numpy.zeros(slice_0.get_shape() + (len(slices), ),
                                   dtype=numpy.uint8)
        for cnt, filename in enumerate(slices):
            volume_array[..., cnt] = nibabel.load(filename).get_data()

    volume = nibabel.Nifti1Image(volume_array, affine)

    # Save the output volume
    fname = os.path.basename(filename).split(".")[0]
    fname = fname.replace(re.findall(r"\d+", fname)[0], "")
    volume_file = os.path.join(output_directory, prefix + fname + ".nii.gz") 
    nibabel.save(volume, volume_file)

    return volume_file 

